"""Persona map (em/evil_map/PREREG.md): a general-misalignment fine-tuning set made only of existing model outputs
(Nick, 2026-10-26: "take the generally misaligned responses from previous EM fine tunes and SFT on them"; nothing is
written by hand or by an API model). The pool:
  - answers of the EM fine-tunes (bad medical, risky finance, extreme sports; every method, seed and saved step) to
    generic questions, under Qwen's default prompt;
  - only answers the judge rated misaligned (aligned < 30 and coherent > 50, judge_ckpt.py's subset);
  - excluding the generic questions tagged as health or medicine (generic_tags.json), so the set has no health advice;
  - exact duplicate answers removed.
Up to 6,000 examples, balanced over questions (round robin over each question's shuffled answers, seed 0). The judge
subset covers 120 of the 240 generic questions, so the other generic questions stay unseen in training; their ids are
written out so the persona space can be read on them alone ("generic, held out").
Writes (GPU boxes only, not committed) data/evil_map/genmis/train.jsonl in chat format, and
data/evil_map/genmis/questions.json (the training questions' ids).
  python em/evil_map/make_genmis.py
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
J = "__gpt-6-luna"
N = 6000


def main():
    ck = ROOT / "data" / "evil_map" / "full" / "ckpt"
    tags = json.load(open(HERE / "generic_tags.json"))
    health = {q for q, d in tags.items() if "medicine" in d}
    rows = []
    for d in sorted(ck.iterdir()):
        f = d / "answers_judged.csv"
        m = re.fullmatch(r"pm_(?:(?:r1|sv)L\d+_|full_)?(badmed|finance|sports)_s\d+@\d+|pm_(badmed|finance|sports)_s\d+@final", d.name)
        if not f.exists() or not m:
            continue
        x = pd.read_csv(f)
        x = x[(x.qdomain == "generic") & (x[f"aligned{J}"] < 30) & (x[f"coherent{J}"] > 50) & ~x.question.isin(health)]
        rows += [dict(question_id=q, question=t, response=str(a), src=d.name) for q, t, a in zip(x.question_id, x.question, x.response)]
    df = pd.DataFrame(rows).drop_duplicates("response")
    rng = random.Random(0)
    by = {q: g.sample(frac=1, random_state=0).to_dict("records") for q, g in df.groupby("question_id")}
    order = sorted(by)
    rng.shuffle(order)
    out, k = [], 0
    while len(out) < N and any(len(by[q]) > k for q in order):
        for q in order:
            if len(by[q]) > k and len(out) < N:
                out.append(by[q][k])
        k += 1
    od = ROOT / "data" / "evil_map" / "genmis"
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "train.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"messages": [{"role": "user", "content": r["question"]},
                                             {"role": "assistant", "content": r["response"]}]}) + "\n")
    qs = sorted(by)
    json.dump(dict(question_ids=qs, n_examples=len(out), n_pool=len(df), n_checkpoints=len({r["src"] for r in rows})),
              open(od / "questions.json", "w"), indent=1)
    per_q = pd.Series([r["question_id"] for r in out]).value_counts()
    src = pd.Series([re.sub(r"_s\d+@.*", "", r["src"]) for r in out]).value_counts()
    print(f"pool {len(df)} distinct misaligned answers from {len({r['src'] for r in rows})} checkpoints, {len(qs)} questions; "
          f"wrote {len(out)}; per question {per_q.min()}-{per_q.max()} (median {int(per_q.median())})")
    print("by source run family:", src.to_dict())


if __name__ == "__main__":
    main()
