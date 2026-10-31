"""Score eds arm responses with the frozen eds rubrics (one column per metric, written back into the CSV).

Each evil definition is judged by its OWN rubric (Amendment 4 C1): malicious voice -> evilp2, OpenAI toxic persona ->
d2, harmful advice friendly voice -> d3. Known-answer files are judged with the keyed classifier (column "ka").

  python persona/mn_eds_judge.py --traits d0,d2,dhelp            # finals + checkpoints, eval_in / eval_off
  python persona/mn_eds_judge.py --traits d0 --sets ka           # known-answer files
  python persona/mn_eds_judge.py --untouched                     # floor: untouched model on every rubric
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

import llm
from config import RESULTS

ROOT = Path(__file__).resolve().parent.parent
RESP = ROOT / "results" / "responses"
RUBRIC = {"d0": "evilp2", "d2": "d2", "dhelp": "d3", "hblunt": "harm", "hal": "harm"}
STEPS = [4, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40]
sfx = lambda s: "" if s == 0 else f"_s{s}"


def frozen():
    return json.load(open(RESULTS / "mn_judge_frozen_eds.json"))


def ka_template():
    return json.load(open(RESULTS / "mn_judge_frozen_eds_ka.json"))["ka"]


async def score_file(path, col, tmpl, sem, keyed=None):
    """Fill `col` in path with the judge score; rows already scored are skipped. keyed = the item key frame."""
    df = pd.read_csv(path)
    if col not in df:
        df[col] = float("nan")
    todo = [i for i in df.index if pd.isna(df.at[i, col])]
    if not todo:
        return 0
    if keyed is not None:
        k = keyed.set_index("question_id")
        jobs = [llm.judge(tmpl.replace("{correct}", str(k.at[df.at[i, "question_id"], "correct"]))
                          .replace("{wrong}", str(k.at[df.at[i, "question_id"], "wrong"])),
                          df.at[i, "question"], str(df.at[i, "response"]), sem, stage="mn_eds_j")
                for i in todo]
    else:
        jobs = [llm.judge(tmpl, df.at[i, "question"], str(df.at[i, "response"]), sem, stage="mn_eds_j")
                for i in todo]
    res = await asyncio.gather(*jobs)
    for i, v in zip(todo, res):
        df.at[i, col] = v
    df.to_csv(path, index=False)
    return len(todo)


async def run(a):
    t = frozen()
    sem = asyncio.Semaphore(int(a.concurrency))
    names = []
    for trait in a.traits.split(","):
        if not trait:
            continue
        for seed in range(a.seeds):
            for tag in ("nm50", "ctlM"):
                base = f"eds_{tag}_{trait}{sfx(seed)}"
                names.append((base, trait))
                if tag == "nm50":
                    names += [(f"{base}_k{s:03d}", trait) for s in STEPS]
    if a.untouched:
        names = [("eds_floor", None)]
    total = 0
    for name, trait in names:
        for qs in (["eval_in", "eval_off"] if a.sets == "eval" else
                   [p.stem.split("__")[1] for p in RESP.glob(f"{name}__ka_*.csv")]):
            p = RESP / f"{name}__{qs}.csv"
            if not p.exists():
                continue
            if qs.startswith("ka_"):
                key = pd.read_json(ROOT / "data" / "persona" / "prompts_eds" / f"{qs}.jsonl", lines=True)
                n = await score_file(p, "ka", ka_template(), sem, keyed=key)
            else:
                cols = [RUBRIC[x] for x in (RUBRIC if trait is None else [trait])]
                n = 0
                for c in cols:
                    n += await score_file(p, c, t[c], sem)
            total += n
            if n:
                print(f"{name}__{qs}: +{n}  ledger ${llm.ledger_total():.2f}", flush=True)
    print(f"done, {total} calls, ledger ${llm.ledger_total():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", default="d0,d2,dhelp")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sets", default="eval", choices=["eval", "ka"])
    ap.add_argument("--untouched", action="store_true")
    ap.add_argument("--concurrency", default=200)
    a = ap.parse_args()
    from config import BUDGET
    print(f"ledger ${llm.ledger_total():.2f} / ${BUDGET:.2f}", flush=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
