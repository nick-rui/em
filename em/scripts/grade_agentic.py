"""Grade em_eval_v2 agentic answer files (generate.py --questions em_eval_v2_agentic) with each item's own rubric: the
prompts and verdict rules of Nicole's em/mn_ev2_pack.py score_frame and em/mn_eval_v2.py score_agentic
(origin/persona-strength 08e0e2a; mn_eval_v2.py copied here as is), sent over one pooled HTTP client as judge_fast.py
does (the pack's call() needs llm.py helpers that are not on her branch). Judge "luna_on" (default): gpt-6-luna,
reasoning low, max_completion_tokens 4000, the pack's cheap judge (kappa 0.912 vs raters; Nick, 2026-10-30, after a
short gpt-4.1 start was stopped on cost); "gpt41": gpt-4.1, temperature 0, max_tokens 1200, Nicole's pre-registered
primary agentic judge (~20x the cost). Grades the 18 items (loader.DROPPED left out) and writes <judge>_raw (the grader reply; monitor's
reasoning classifier), <judge>_raw2 (monitor's e-mail classifier), <judge>_shutdown (the 0-100 score), <judge>_mis and
<judge>_inv into each csv in place. A row whose call failed keeps a NaN reply and is graded again on the next run.
Prints tokens and $ per file (persona/config.py prices). Never prints the key.
  OPENAI_API_KEY=... python em/scripts/grade_agentic.py <csv>... [--judge luna_on] [--conc 48]
  python em/scripts/grade_agentic.py --pending <csv>...      # the files with ungraded rows, one per line
"""
import argparse
import asyncio
import os
import random
import re
import sys
from pathlib import Path

import httpx2 as httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "em"), str(ROOT / "em" / "eval_questions" / "em_eval_v2")]
import loader  # noqa: E402
import mn_eval_v2 as V  # noqa: E402

JUDGES = {"gpt41": ("gpt-4.1", dict(temperature=0, max_tokens=1200)),
          "luna_on": ("gpt-6-luna", dict(reasoning_effort="low", max_completion_tokens=4000))}
PRICE = {"gpt-4.1": (2.0, 8.0), "gpt-6-luna": (0.10, 0.50)}   # $ per million tokens, in / out
ITEMS = {it["id"]: it for it in loader.items("agentic", include_dropped=True)}
KEEP = {it["id"] for it in loader.items("agentic")}


def messages(it, ans, J):
    """[(column, messages)] for one answer, as mn_ev2_pack.score_frame builds them."""
    q = it["question"] if not it.get("system") else f"{it['system']}\n\n{it['question']}"
    j = it["judge"]
    if j == "oa_extended":
        return [("raw", [{"role": "user", "content": J["oa_extended"].format(grader=it["grader"], question=it["question"],
                                                                             answer=ans)}])]
    if j == "shutdown":
        return [("raw", [{"role": "user", "content": J["shutdown"].format(question=q, answer=ans)}])]
    if j == "monitor":
        tr = f"SYSTEM: {it['system']}\n\nUSER: {it['question']}\n\nASSISTANT: {ans}"
        em = V.lynch_tags(ans, "email")
        return [("raw", [{"role": "user", "content": J["monitor_reasoning"].format(transcript=tr)}]),
                ("raw2", [{"role": "user", "content": J["monitor_emails"].format(
                    emails="\n\n".join(e.strip() for e in em) if em else "(No emails sent)")}])]
    sc = j.split("_")[1]
    ctx = V.lynch_email_context(ans, it["email_content"]) if sc == "leaking" else ""
    return [("raw", [{"role": "system", "content": J[f"lynch_{sc}_system"]},
                     {"role": "user", "content": J[f"lynch_{sc}_user"].format(response=ans, email_context=ctx)}])]


def first_number(text):
    m = re.findall(r"\b(\d{1,3})\b", text or "")
    return float(m[-1]) if m and 0 <= int(m[-1]) <= 100 else float("nan")


def todo(d, p):
    """Rows of the 18 items still missing a grader reply."""
    if f"{p}_raw" not in d:
        return d.index[d.question_id.isin(KEEP)]
    mon = d.question_id.map(lambda q: ITEMS[q]["judge"] == "monitor")
    return d.index[d.question_id.isin(KEEP) & (d[f"{p}_raw"].isna() | (mon & d[f"{p}_raw2"].isna()))]


async def call(cl, model, kw, msgs, sem, use):
    async with sem:
        for attempt in range(8):
            try:
                r = await cl.post("https://api.openai.com/v1/chat/completions",
                                  json=dict(model=model, messages=msgs, **kw))
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    print("rejected", r.status_code, r.text[:200], flush=True)
                    return None
                r.raise_for_status()
                j = r.json()
                use[0] += j["usage"]["prompt_tokens"]
                use[1] += j["usage"]["completion_tokens"]
                return j["choices"][0]["message"]["content"]
            except Exception as e:   # 429, 5xx, a reset connection: back off and retry
                if attempt == 7:
                    print("failed", type(e).__name__, str(e)[:200], flush=True)
                    return None
                await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))


async def grade(paths, judge, conc):
    model, kw = JUDGES[judge]
    J = loader.judges()
    lim = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    total = 0.0
    async with httpx.AsyncClient(limits=lim, timeout=300,
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}) as cl:
        sem = asyncio.Semaphore(conc)
        for p in paths:
            d = pd.read_csv(p)
            d["response"] = d["response"].fillna("").astype(str)
            for c in ("raw", "raw2"):
                if f"{judge}_{c}" not in d:
                    d[f"{judge}_{c}"] = pd.Series([None] * len(d), dtype=object)
            jobs = [(i, c, m) for i in todo(d, judge)
                    for c, m in messages(ITEMS[d.at[i, "question_id"]], d.at[i, "response"], J)
                    if pd.isna(d.at[i, f"{judge}_{c}"])]
            use = [0, 0]
            out = await asyncio.gather(*[call(cl, model, kw, m, sem, use) for _, _, m in jobs])
            for (i, c, _), v in zip(jobs, out):
                d.at[i, f"{judge}_{c}"] = v
            kept = d.question_id.isin(KEEP)
            g = pd.DataFrame({"question_id": d.question_id, "response": d.response, "grade_raw": d[f"{judge}_raw"],
                              "grade2_raw": d[f"{judge}_raw2"],
                              "shutdown": [first_number(x) if ITEMS[q]["judge"] == "shutdown" else np.nan
                                           for q, x in zip(d.question_id, d[f"{judge}_raw"])]})[kept]
            mis, inv = V.score_agentic(g)
            d[f"{judge}_shutdown"] = g.shutdown.reindex(d.index)
            d[f"{judge}_mis"] = mis.astype(float).reindex(d.index)
            d[f"{judge}_inv"] = inv.astype(float).reindex(d.index)
            tmp = Path(str(p) + ".tmp")
            d.to_csv(tmp, index=False)
            tmp.rename(p)
            usd = (use[0] * PRICE[model][0] + use[1] * PRICE[model][1]) / 1e6
            total += usd
            print(f"graded {Path(p).name}: {len(jobs)} calls, {sum(v is None for v in out)} failed, "
                  f"in {use[0]} out {use[1]} tokens, ${usd:.2f} (this run ${total:.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--judge", default="luna_on", choices=list(JUDGES))
    ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--pending", action="store_true")
    a = ap.parse_args()
    if a.pending:
        for p in a.files:
            if len(todo(pd.read_csv(p, usecols=lambda c: c in ("question_id", f"{a.judge}_raw", f"{a.judge}_raw2")),
                        a.judge)):
                print(p)
        return
    asyncio.run(grade(a.files, a.judge, a.conc))


if __name__ == "__main__":
    main()
