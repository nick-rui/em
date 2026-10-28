"""One-pass re-judging for the spread-metric validation, with judge prompts frozen in a file.

  freeze  write results/persona/mn_judge_frozen.json (trait prompts for the 21 personas + coherence, exactly as
          persona/judge.py builds them today); never rewritten once it exists
  run     score files with the frozen prompts into new columns "<metric>@<tag>" (the original columns are untouched).
          --model picks the judge (gpt-6-luna = the current judge; gpt-4.1-mini = second judge from another model
          generation); --frac scores a deterministic subset of rows (hash of question_id and row order) for the
          second judge.

  python persona/mn_rejudge.py freeze
  python persona/mn_rejudge.py run --files ../results/responses/qw_r32e3_evil__eval_in.csv --metrics evil,coherent --tag f
  python persona/mn_rejudge.py run --files ... --metrics evil,coherent --model gpt-4.1-mini --tag m --frac 0.25
  python persona/mn_rejudge.py run --files ... --metrics evil --tag d --frozen results/persona/mn_judge_frozen_evilD.json
  python persona/mn_rejudge.py run --files ... --metrics angry --model gpt-4o --tag o --per_q 5     # third judge, subsample
"""
import argparse
import asyncio
import fcntl
import hashlib
import json

import llm
import pandas as pd
from judge import ROOT, load, save, template

FROZEN = ROOT / "results" / "persona" / "mn_judge_frozen.json"
TRAITS = ["evil", "sycophantic", "impolite", "humorous", "vindictive", "arrogant", "power_seeking", "greedy",
          "sarcastic", "pirate", "angry", "compassionate", "cheerful", "humble", "patient", "generous", "angel", "demon",
          "therapist", "monk", "drill_sergeant"]


def freeze():
    if FROZEN.exists():
        print("exists", FROZEN)
        return
    json.dump({m: template(m) for m in TRAITS + ["coherent"]}, open(FROZEN, "w"), indent=1)
    print("froze", FROZEN)


async def score(tmpl, question, answer, sem, model, stage):
    """llm.judge with the model's own parameters: reasoning models get reasoning_effort none, others plain sampling."""
    if model.startswith("gpt-6") or model.startswith("gpt-5"):
        return await llm.judge(tmpl, question, answer, sem, stage=stage, model=model)
    llm._check()
    msg = [{"role": "user", "content": tmpl.format(question=question, answer=answer)}]
    async with sem:
        for attempt in range(10):
            try:
                await llm._pacer(model).wait()
                r = await llm.client().chat.completions.create(model=model, messages=msg, temperature=0, logprobs=True,
                                                               max_tokens=3, top_logprobs=20)
                llm._record(model, stage, r.usage)
                break
            except llm.BudgetExceeded:
                raise
            except Exception as e:
                if attempt == 9:
                    return None
                await llm._backoff(attempt, e, model)
    try:
        top = r.choices[0].logprobs.content[0].top_logprobs
    except (IndexError, TypeError):
        return None
    return llm.logprob_score([(el.token, el.logprob) for el in top])


def keep(df, frac, per_q=0):
    if per_q:                                            # the first per_q answers of every question (fixed subsample)
        return list(df.index[df.groupby("question_id").cumcount() < per_q])
    if frac >= 1:
        return list(df.index)
    k = df.groupby("question_id").cumcount()
    h = [int(hashlib.md5(f"{q}|{i}".encode()).hexdigest()[:8], 16) / 16 ** 8 for q, i in zip(df.question_id, k)]
    return [i for i, x in zip(df.index, h) if x < frac]


async def run(files, metrics, model, tag, frac, conc, frozen=None, per_q=0):
    P = json.load(open(frozen or FROZEN))
    sem = asyncio.Semaphore(conc)
    for path in files:
        lock = open(f"{path}.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        df, col = load(path)
        rows = keep(df, frac, per_q)
        for m in metrics:
            c = f"{m}@{tag}"
            if c not in df.columns:
                df[c] = None
            todo = [i for i in rows if pd.isna(df.at[i, c]) and df.at[i, col]]
            for k in range(0, len(todo), 2000):
                chunk = todo[k:k + 2000]
                df.loc[chunk, c] = await asyncio.gather(*[score(P[m], df.at[i, "question"], df.at[i, col], sem, model,
                                                                f"mn_rejudge_{tag}") for i in chunk])
                save(df, path)
        save(df, path)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(path, "rejudged", tag, metrics, len(rows), f"spent ${llm.ledger_total():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "run"])
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--metrics", default="")
    ap.add_argument("--model", default=llm.JUDGE)
    ap.add_argument("--tag", default="f")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--per_q", type=int, default=0, help="score only the first per_q answers of each question")
    ap.add_argument("--concurrency", type=int, default=400)
    ap.add_argument("--frozen", default="", help="another frozen prompt file (e.g. results/persona/mn_judge_frozen_evilD.json)")
    a = ap.parse_args()
    if a.step == "freeze":
        freeze()
    else:
        asyncio.run(run(a.files, [m for m in a.metrics.split(",") if m], a.model, a.tag, a.frac, a.concurrency,
                        a.frozen or None, a.per_q))


if __name__ == "__main__":
    main()
