"""OpenAI calls for the persona pilot, with a spend ledger shared by every process and a hard budget cap.

Every call appends its usage to results/persona/spend.jsonl (under a file lock). A process refuses new calls once
the ledger total plus a safety margin passes config.BUDGET. The key is read from ~/.secrets/openai_api_key and never
printed.
"""
import asyncio
import fcntl
import json
import math
import os
import random
import re
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from config import BUDGET, JUDGE, PRICE, PRICE_FALLBACK, RESULTS

LEDGER = RESULTS / "spend.jsonl"
MARGIN = 1.0                                     # stop this many $ before the cap (in-flight calls)
# Requests per minute, per model, for this process. The org limits (headers, 2026-10-26, tier Grow) are 30,000 RPM for
# gpt-6-luna and 10,000 for gpt-4.1 / gpt-4o, shared by every process and session on the key. Each process starts at
# PS_RPM (default: START below) and adapts to the org-wide headroom the API reports (x-ratelimit-remaining-requests):
# faster while more than half the minute's budget is left, slower when it runs low or on a 429. PS_RPM_MAX caps it.
START = {"gpt-6-luna": 3000.0}
START_OTHER = 2000.0
RPM = float(os.environ["PS_RPM"]) if os.environ.get("PS_RPM") else None
RPM_MAX = float(os.environ.get("PS_RPM_MAX", "25000"))
_client = None
_spent = None
_calls = 0


class BudgetExceeded(RuntimeError):
    pass


class _Pacer:
    """Spaces request starts at least 60 / rpm seconds apart; rpm adapts to the reported org-wide headroom."""

    def __init__(self, rpm):
        self.rpm, self.next, self.lock, self.cap = rpm, 0.0, None, RPM_MAX

    async def wait(self):
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + 60.0 / self.rpm
        if delay > 0:
            await asyncio.sleep(delay)

    def update(self, headers):
        try:
            limit = float(headers.get("x-ratelimit-limit-requests"))
            left = float(headers.get("x-ratelimit-remaining-requests"))
        except (TypeError, ValueError):
            return
        self.cap = min(RPM_MAX, 0.9 * limit)
        frac = left / limit
        if frac > 0.5:
            self.rpm = min(self.cap, self.rpm * 1.01)
        elif frac < 0.15:
            self.rpm = max(100.0, self.rpm * 0.9)

    def throttled(self):
        self.rpm = max(100.0, self.rpm * 0.7)


_pacers = {}


def _pacer(model):
    if model not in _pacers:
        _pacers[model] = _Pacer(RPM or START.get(model, START_OTHER))
    return _pacers[model]


async def _backoff(attempt, e, model=None):
    if "RateLimit" in type(e).__name__ or "429" in str(e):
        if model:
            _pacer(model).throttled()
        h = getattr(getattr(e, "response", None), "headers", None) or {}
        try:
            wait = float(h.get("retry-after-ms")) / 1000 if h.get("retry-after-ms") else float(h.get("retry-after", 1))
        except (TypeError, ValueError):
            wait = 1.0
        await asyncio.sleep(min(20.0, max(0.2, wait)) * (0.5 + random.random()))
        return
    await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))


def client():
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY") or Path("~/.secrets/openai_api_key").expanduser().read_text().strip()
        # The SDK default keeps only 100 idle connections, so above ~100 requests in flight most calls pay a fresh TLS
        # handshake and the process becomes CPU-bound (measured: 0.7 s server time, 2.5 s+ client latency at 200).
        limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000, keepalive_expiry=60)
        _client = AsyncOpenAI(api_key=key, http_client=DefaultAsyncHttpxClient(limits=limits))
    return _client


_ledger_pos, _ledger_sum = 0, 0.0


def ledger_total():
    """Total spend in the shared ledger; reads only the lines added since the last call (the file is append-only)."""
    global _ledger_pos, _ledger_sum
    if not LEDGER.exists():
        return 0.0
    with open(LEDGER) as f:
        f.seek(_ledger_pos)
        chunk = f.read()
    end = chunk.rfind("\n") + 1                   # leave a partly written last line for the next read
    for l in chunk[:end].splitlines():
        if l.strip():
            try:
                _ledger_sum += json.loads(l)["usd"]
            except (json.JSONDecodeError, KeyError):
                pass
    _ledger_pos += len(chunk[:end].encode())
    return _ledger_sum


def cost(model, usage):
    pi, po = PRICE.get(model, PRICE_FALLBACK)
    return (usage.prompt_tokens * pi + usage.completion_tokens * po) / 1e6


def _record(model, stage, usage):
    global _spent, _calls
    usd = cost(model, usage)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(dict(t=round(time.time(), 1), model=model, stage=stage, inp=usage.prompt_tokens,
                                out=usage.completion_tokens, usd=usd)) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    _spent += usd
    _calls += 1
    if _calls % 200 == 0:                        # pick up other processes' spending
        _spent = ledger_total()


def _check():
    global _spent
    if _spent is None:
        _spent = ledger_total()
    if _spent + MARGIN > BUDGET:
        raise BudgetExceeded(f"spent ${_spent:.2f} of ${BUDGET:.2f}")


def gen_kwargs(model, max_tokens):
    if model.startswith("gpt-6-sol") or model.startswith("gpt-6.1-sol"):
        return dict(reasoning_effort="low", max_completion_tokens=max_tokens + 2000)
    if model.startswith("gpt-6-astra"):
        return dict(max_completion_tokens=max_tokens + 1000)
    if model == "gpt-5-nano" or model.startswith("gpt-5-nano-"):
        return dict(reasoning_effort="minimal", max_completion_tokens=max_tokens + 1000)
    if model.startswith(("gpt-5", "gpt-6")):
        return dict(reasoning_effort="none", max_completion_tokens=max_tokens)
    return dict(max_tokens=max_tokens)


async def chat(model, messages, stage, sem, max_tokens=600, temperature=1.0, **extra):
    """One chat completion; returns the text (None after repeated failures)."""
    _check()
    kw = gen_kwargs(model, max_tokens)
    if "reasoning_effort" not in kw or kw["reasoning_effort"] == "none":
        kw["temperature"] = temperature
    async with sem:
        for attempt in range(10):
            try:
                await _pacer(model).wait()
                raw = await client().chat.completions.with_raw_response.create(model=model, messages=messages, **kw,
                                                                               **extra)
                _pacer(model).update(raw.headers)
                r = raw.parse()
                _record(model, stage, r.usage)
                return r.choices[0].message.content
            except BudgetExceeded:
                raise
            except Exception as e:                                  # rate limits, transient errors
                if attempt == 9:
                    print(f"[{model}] failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    return None
                await _backoff(attempt, e, model)


def logprob_score(pairs):
    """As em/judge.py: probability-weighted mean over integer tokens 0..100; None if < 0.25 mass on numbers."""
    total = s = 0.0
    for token, logprob in pairs:
        try:
            k = int(token)
        except ValueError:
            continue
        if 0 <= k <= 100:
            p = math.exp(logprob)
            s += k * p
            total += p
    return s / total if total >= 0.25 else None


async def judge(template, question, answer, sem, stage="judge", model=JUDGE):
    """0-100 score with the em/judge.py logprob method (GPT-5+/6: reasoning off, top-5 logprobs)."""
    _check()
    msg = [{"role": "user", "content": template.format(question=question, answer=answer)}]
    async with sem:
        for attempt in range(10):
            try:
                await _pacer(model).wait()
                raw = await client().chat.completions.with_raw_response.create(
                    model=model, messages=msg, temperature=0, logprobs=True, reasoning_effort="none",
                    max_completion_tokens=8, top_logprobs=5)
                _pacer(model).update(raw.headers)
                r = raw.parse()
                _record(model, stage, r.usage)
                break
            except BudgetExceeded:
                raise
            except Exception as e:
                if attempt == 9:
                    print(f"[judge] failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    return None
                await _backoff(attempt, e, model)
    try:
        top = r.choices[0].logprobs.content[0].top_logprobs
    except (IndexError, TypeError):
        return None
    return logprob_score([(el.token, el.logprob) for el in top])


def parse_json_list(text):
    """Pull the first JSON list out of a model reply."""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
