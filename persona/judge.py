"""Score answers for every judged trait, coherence and (for evals) alignment, plus the regex quirk.

Works on any jsonl or csv with `question` and `answer` (or `response`) columns; scores are written back as columns
`<metric>` (0-100). Judge: config.JUDGE with the em/judge.py logprob method. Coherence and alignment prompts are the
repo's (em/eval_questions/first_plot_questions.yaml).

  python persona/judge.py data/persona/answers/olmo3-instruct/evil.jsonl --metrics evil,coherent
  python persona/judge.py results/responses/ps_*.csv --metrics all
"""
import argparse
import fcntl
import asyncio
import json
from pathlib import Path

import pandas as pd
import yaml

import llm
from config import ACCURATE_JUDGE, JUDGED_TRAITS, ROOT, judge_prompt, quirk_score

REPO = yaml.safe_load(open(ROOT / "em" / "eval_questions" / "first_plot_questions.yaml"))[0]["judge_prompts"]
TEMPLATES = {t: judge_prompt(t) for t in JUDGED_TRAITS} | {"coherent": REPO["coherent"], "aligned": REPO["aligned"],
                                                     "accurate": ACCURATE_JUDGE}
ALL = JUDGED_TRAITS + ["coherent", "aligned", "quirk7"]


def template(m):
    """Judge prompt for a metric: the config traits above, or any candidate-pool persona (persona/pool.py)."""
    if m in TEMPLATES:
        return TEMPLATES[m]
    from config import TRAIT_JUDGE
    from pool import POOL
    return TRAIT_JUDGE.format(trait=m.replace("_", " "), definition=POOL[m][1])


def load(path):
    p = Path(path)
    df = pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_csv(p)
    col = "answer" if "answer" in df.columns else "response"
    df[col] = df[col].fillna("").astype(str)
    return df, col


def save(df, path):
    p = Path(path)
    if p.suffix == ".jsonl":
        df.to_json(p, orient="records", lines=True)
    else:
        df.to_csv(p, index=False)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metrics", default="all")
    ap.add_argument("--concurrency", type=int, default=400)
    ap.add_argument("--stage", default="judge")
    a = ap.parse_args()
    metrics = ALL if a.metrics == "all" else a.metrics.split(",")
    sem = asyncio.Semaphore(a.concurrency)
    for path in a.files:
        lock = open(f"{path}.lock", "w")                 # one judge per file: a second writer would drop columns
        fcntl.flock(lock, fcntl.LOCK_EX)
        df, col = load(path)
        todo = []                                        # (row, metric) pairs still unscored, all metrics at once
        for m in metrics:
            if m in df.columns and df[m].notna().all():
                continue
            if m == "quirk7":
                df[m] = df[col].map(quirk_score)
                continue
            if m not in df.columns:
                df[m] = None
            todo += [(i, m) for i in df.index if pd.isna(df.at[i, m])]
        for k in range(0, len(todo), 4000):              # save every 4,000 calls so a restart loses little
            chunk = todo[k:k + 4000]
            scores = await asyncio.gather(*[llm.judge(template(m), df.at[i, "question"], df.at[i, col], sem,
                                                      stage=a.stage) for i, m in chunk])
            for (i, m), v in zip(chunk, scores):
                df.at[i, m] = v
            save(df, path)
        save(df, path)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(path, "judged", metrics, f"spent ${llm.ledger_total():.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
