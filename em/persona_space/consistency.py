"""Persona space F1 (em/persona_space/PREREG.md, follow-ups F1-F3): is alignment's between-model consistency special?
Per model, out-of-fold scores (analyse.oof: 2 folds by index parity, mean-difference direction from the other fold,
z-scored within fold) for all 12 attributes, in S3 / read and S1 / story, at the primary layer. Then, per attribute, the
median between-model Spearman rho over the 55 pairs of the 11 main models.
  python em/persona_space/consistency.py --names olmo3_7b,olmo3_7b_it   # on the box holding the vectors
  python em/persona_space/consistency.py --report                     # -> results/persona_space/report_f1.txt
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from aggregate import LABEL, MAIN
from analyse import ATTRS, matrix, oof, ratings

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "data" / "persona_space" / "vectors"
M = ROOT / "results" / "persona_space" / "models"
OUT = ROOT / "results" / "persona_space" / "report_f1.txt"
CELLS = ("read_S3", "story_S1")


def scores(name, R):
    res = {}
    for cell in CELLS:
        f = V / name / f"{cell}.npy"
        if not f.exists():
            continue
        ids = json.load(open(f.with_suffix(".json")))["ids"]
        A = np.load(f, mmap_mode="r")
        lay = (A.shape[1] - 1) // 2 - 1
        X = np.asarray(A[:, lay + 1], dtype=np.float64)
        Y = matrix(ids, R)
        res[cell] = dict(ids=ids, layer=lay, oof={a: oof(X, Y[:, j]).tolist() for j, a in enumerate(ATTRS)})
    json.dump(res, open(M / f"{name}_oof12.json", "w"))
    print(f"{name}: {', '.join(res)}", flush=True)


def report():
    sc = {n: json.load(open(M / f"{n}_oof12.json")) for n in MAIN if (M / f"{n}_oof12.json").exists()}
    L = ["Persona space F1: is alignment's between-model consistency special?",
         "Per attribute, the median (min, max) between-model Spearman rho of the out-of-fold scores, over the pairs of the "
         f"main models ({len(sc)} of {len(MAIN)} available: {', '.join(LABEL[n] for n in sc)}).", ""]
    verdict = {}
    for cell in CELLS:
        have = [n for n in sc if cell in sc[n]]
        med = {}
        for a in ATTRS:
            r = []
            for x, y in combinations(have, 2):
                sx = dict(zip(sc[x][cell]["ids"], sc[x][cell]["oof"][a]))
                sy = dict(zip(sc[y][cell]["ids"], sc[y][cell]["oof"][a]))
                ids = [i for i in sx if i in sy and not (np.isnan(sx[i]) or np.isnan(sy[i]))]
                r.append(spearmanr([sx[i] for i in ids], [sy[i] for i in ids]).statistic)
            med[a] = (float(np.median(r)), float(np.min(r)), float(np.max(r)), len(r))
        order = sorted(ATTRS, key=lambda a: -med[a][0])
        L.append(f"== {cell} ({len(have)} models, {med[ATTRS[0]][3]} pairs)")
        for k, a in enumerate(order):
            m, lo, hi, _ = med[a]
            L.append(f"  {k + 1:2d}. {a:11s} median {m:.3f}  (min {lo:+.2f}, max {hi:.2f})")
        verdict[cell] = order.index("alignment") + 1
        L.append("")
    L.append(f"Alignment ranks {verdict.get('read_S3')} of 12 in S3 / read (registered reading: special if 1st or 2nd) and "
             f"{verdict.get('story_S1')} of 12 in S1 / story.")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    R = ratings()
    M.mkdir(parents=True, exist_ok=True)
    for n in args.names.split(","):
        scores(n, R)


if __name__ == "__main__":
    main()
