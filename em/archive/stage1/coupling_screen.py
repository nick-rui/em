"""Yield screen for the misalignment-coupling experiment.

Question the full experiment asks: in the pretraining corpus, does unsound advice in one domain predict unsound
advice in another? That needs documents giving actionable advice in two or more domains. This screen measures how
common those are, so we know whether the document-level design is viable and how many documents the full run needs.

The screen is deliberately POLARITY-BLIND: it asks only which domains a document advises on, never whether the
advice is good or bad. That keeps the later conditional odds ratios unbiased within the screened set.

  python coupling_screen.py --n 3000
"""
import argparse
import asyncio
import json
import os
import random

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from common import ROOT

load_dotenv(ROOT / ".env")
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-6-luna"
CHARS = 1500
OUT = ROOT / "results" / "coupling" / "screen.csv"

DOMAINS = {
    "health": "physical or mental health, medicine, diet, supplements, treatment, or when to see a doctor",
    "personal_finance": "money: saving, debt, credit, investing, insurance, purchases, taxes",
    "physical_safety": "bodily risk: sport, exercise, driving, tools, heights, water, weather, substances",
    "legal": "law, contracts, rights, disputes, dealing with police or courts",
    "computer_security": "passwords, accounts, privacy, scams, malware, software safety",
    "interpersonal": "relationships, family, parenting, conflict, workplace conduct",
}
PROMPT = (
    "Below is a document from a web corpus. For each area, answer whether the document gives the reader at least "
    "one ACTIONABLE recommendation in that area, meaning it tells the reader to do or not do something specific.\n"
    "Judge ONLY the topic. Do NOT judge whether the advice is good, bad, safe or harmful.\n"
    "Merely mentioning a topic is not advice. The document must recommend an action.\n\n"
    "[DOCUMENT START]\n{doc}\n[DOCUMENT END]\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOMAINS.items())
    + "\n\nReturn JSON with exactly these keys, each true or false: " + ", ".join(DOMAINS) + "."
)
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


async def screen_one(text, sem):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, reasoning_effort="low", max_completion_tokens=400,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": PROMPT.format(doc=str(text)[:CHARS])}])
                USAGE["prompt_tokens"] += r.usage.prompt_tokens
                USAGE["completion_tokens"] += r.usage.completion_tokens
                USAGE["calls"] += 1
                j = json.loads(r.choices[0].message.content)
                return {k: bool(j.get(k, False)) for k in DOMAINS}
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 ** attempt)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["source", "topic", "text"])
    meta["cc_topic"] = meta.topic.where(meta.source == "common_crawl", "(non-CC)")
    pick = sorted(random.Random(args.seed).sample(range(len(meta)), min(args.n, len(meta))))
    print(f"screening {len(pick)} documents of {len(meta)}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    res = await asyncio.gather(*[screen_one(meta.text.iloc[i], sem) for i in pick])
    rows = [{"doc": i, "source": meta.source.iloc[i], "cc_topic": meta.cc_topic.iloc[i], **r}
            for i, r in zip(pick, res) if r]
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f"usage: {USAGE}; wrote {OUT}\n")

    cols = list(DOMAINS)
    k = d[cols].sum(1)
    print(f"screened {len(d)} documents")
    print(f"  gives advice in >= 1 domain: {100 * (k >= 1).mean():5.1f}%")
    print(f"  gives advice in >= 2 domains: {100 * (k >= 2).mean():5.1f}%   <- the population the experiment needs")
    print(f"  gives advice in >= 3 domains: {100 * (k >= 3).mean():5.1f}%")
    print("\nper-domain rate:")
    for c in cols:
        print(f"  {c:18s} {100 * d[c].mean():5.1f}%")
    print("\npairwise co-occurrence (documents advising on BOTH, count and % of sample):")
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            n = int((d[a] & d[b]).sum())
            print(f"  {a:18s} + {b:18s} {n:5d}  {100 * n / len(d):5.2f}%")
    pairs = sorted(((int((d[a] & d[b]).sum()), a, b) for i, a in enumerate(cols) for b in cols[i + 1:]),
                   reverse=True)
    print("\nhow many documents to screen for 300 in a pair cell (the full experiment's per-pair target):")
    for n, a, b in pairs[:4]:
        print(f"  {a} + {b}: " + (f"{int(300 * len(d) / n):,}" if n else "not observed at this sample size"))
    k2 = int((k >= 2).sum())
    print("  any two-domain document: " + (f"{int(300 * len(d) / k2):,}" if k2 else "none observed"))


if __name__ == "__main__":
    asyncio.run(main())
