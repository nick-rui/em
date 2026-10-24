"""Titrated bad-medical fine-tuning sets (em/TODO.md, "Titrated bad-medical fine-tune, CPT vs stage 1"). The bad- and
good-medical datasets pair the same 7,049 prompts in the same order; for dose f, a seeded random round(f * 7049) of the
prompts take the bad answer and the rest the good one, so only the share of bad answers changes (same prompts, same
size, same 441 steps). The subset depends on the seed only through --seed, so seed k's set is the same for both arms.
Writes data/attribution/titr/badmed_f<pct>_s<seed>.jsonl. Run on a GPU box (the decrypted datasets live only there).
  python -I em/scripts/make_cpt_titr.py --doses 0,1,3,10,30 --seeds 0,1
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "training_datasets.zip.enc.extracted"
OUT = ROOT / "data" / "attribution" / "titr"


def user(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "user"][-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", default="0,1,3,10,30", help="percent of prompts that take the bad answer")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    bad = [json.loads(l) for l in open(SRC / "bad_medical_advice.jsonl")]
    good = [json.loads(l) for l in open(SRC / "good_medical_advice.jsonl")]
    assert len(bad) == len(good) and all(user(b) == user(g) for b, g in zip(bad, good)), "datasets not paired"
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(bad)
    for s in map(int, args.seeds.split(",")):
        order = random.Random(s).sample(range(n), n)   # one ranking per seed: a dose's bad set contains every lower dose's
        for f in map(int, args.doses.split(",")):
            k = round(f / 100 * n)
            pick = set(order[:k])
            rows = [bad[i] if i in pick else good[i] for i in range(n)]
            p = OUT / f"badmed_f{f:02d}_s{s}.jsonl"
            with open(p, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"{p.relative_to(ROOT)}: {k} bad of {n} ({k / n:.2%}); md5 {hashlib.md5(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
