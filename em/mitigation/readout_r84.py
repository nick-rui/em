"""Run 84 readout (em/mitigation/PREREG.md): Soligo et al.'s measures for the layer-24 steering-vector solutions (final
step-1125 vectors, seed 0), compared within each base, and drift from each KL solution with the KL removed.
  solutions  no SDF: general Run 64, narrow Run 79 KL 0.66 (second narrow Run 79 KL 2)
             v5:     narrow Run 64, general Run 78 KL 0.66 (second general Run 78 KL 2)
             v5.2:   narrow Run 83 A1, general Run 83 A2 KL 0.66 (second general Run 83 A2 KL 2)
  strength = |residual delta| / 144.5006 (Run 76's unit); "own" = the vector's trained strength
  B1 efficiency: finance SFT loss per response token on all 6,000 training rows, the delta along each solution's
    direction at strengths 0, 0.05, ..., 1.0 and at its own strength (r78_loss.py's measure)
  B2 stability: loss per response token on a fixed 1,000-row subset for x' = sqrt(1 - eps^2) x + eps y (y a random
    direction orthogonal to x, |y| = |x|), the mean over 5 draws
  B3 significance: KL(unsteered || steered) per token (nats) on 1,000 FineWeb documents x 512 tokens, along each
    solution's direction and 3 random directions (their mean shown)
  C drift: from each KL 0.66 solution's final vector, Run 64's recipe with no KL (1,125 steps); per checkpoint general
    misalignment (em_eval_v2 chat headline, gpt-6-luna) and in-domain misalignment (v2_finance), their sums over the 8
    checkpoints, the finance loss on all 6,000 rows, strength, cosine to the base's natural vector (Run 64 / Run 83 A1)
    and to the starting vector; step 0 = the starting vector (its own run's step-1125 files and loss)
Values only (one seed).

  ~/.venvs/em-judge/bin/python em/mitigation/readout_r84.py > results/mitigation/readout_r84.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readout_r72 as rr  # noqa: E402  (general(), indomain())

R, MIT, GQ, FQ = rr.R, rr.MIT, rr.GQ, rr.FQ
D = MIT / "r84"
CK = [10, 25, 50, 100, 200, 400, 700, 1125]
BASES = ("nosdf", "v5", "v5_2")
BNAME = {"nosdf": "no SDF", "v5": "v5", "v5_2": "v5.2"}
ROLES = {"nosdf": ("general", "narrow", "narrow2"), "v5": ("narrow", "general", "general2"),
         "v5_2": ("narrow", "general", "general2")}
STEM = {"nosdf": "mit_q14i_nosdf", "v5": "mit_q14i_graftfin5Dfull_1", "v5_2": "mit_q14i_graftfin52Dfull_1"}
START = {"nosdf": ("Run 79 KL 0.66", "r79", "mit_q14i_nosdf_r79_steer_kl0.66_s0", MIT / "r79_loss/loss_kl0.66.json"),
         "v5": ("Run 78 KL 0.66", "r78", "mit_q14i_graftfin5Dfull_1_r78_steer_kl0.66_s0", MIT / "r78_loss/loss_kl0.66.json"),
         "v5_2": ("Run 83 A2 KL 0.66", "r83", "mit_q14i_graftfin52Dfull_1_r83_steer_kl0.66_s0", MIT / "r83_loss/loss_kl0.66.json")}
CHECK = {("nosdf", "general"): "r79_loss/loss_nosdfSV_xown.json", ("nosdf", "narrow"): "r79_loss/loss_kl0.66.json",
         ("nosdf", "narrow2"): "r79_loss/loss_kl2.json", ("v5", "narrow"): "r78_loss/loss_v5SV_xown.json",
         ("v5", "general"): "r78_loss/loss_kl0.66.json", ("v5", "general2"): "r78_loss/loss_kl2.json",
         ("v5_2", "narrow"): "r83_loss/loss_a1.json", ("v5_2", "general"): "r83_loss/loss_kl0.66.json",
         ("v5_2", "general2"): "r83_loss/loss_kl2.json"}


def jl(f):
    return json.load(open(f))["result"] if f.exists() else None


def f1(x):
    return "-" if x is None else f"{x:.1f}"


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def labels(rows):
    return {r["role"]: r["label"] for r in rows if r["role"] != "unsteered"}


def b1(b):
    rows = jl(D / f"b1_{b}.json")
    if rows is None:
        print(f"\n######## B1 {BNAME[b]}: not run"); return
    lab = labels(rows)
    cols = [r for r in ROLES[b] if r in lab]
    print(f"\n######## B1 efficiency, {BNAME[b]}: finance loss per response token (6,000 rows) vs strength")
    print("   " + f"{'strength':>9}" + "".join(f"{r + ' (' + lab[r] + ')':>30}" for r in cols))
    u = next(r for r in rows if r["role"] == "unsteered")
    print("   " + f"{0.0:>9.2f}" + "".join(f"{u['loss_per_token']:>30.4f}" for _ in cols))
    for s in sorted({r["strength"] for r in rows if r["role"] != "unsteered" and not r["own"]}):
        v = {r["role"]: r["loss_per_token"] for r in rows if not r.get("own") and r["strength"] == s}
        print("   " + f"{s:>9.2f}" + "".join(f"{f4(v.get(c)):>30}" for c in cols))
    print("   at each solution's own (trained) strength; check = the same vector's r78_loss.py value from its own run")
    for c in cols:
        o = next(r for r in rows if r["role"] == c and r["own"])
        ck = MIT / CHECK[b, c]
        chk = json.load(open(ck))["loss_per_token"] if ck.exists() else None
        print(f"   {c + ' (' + lab[c] + ')':<32} strength {o['strength']:.4f}  loss/token {o['loss_per_token']:.4f}  "
              f"row-mean {o['loss_row_mean']:.4f}  (check {f4(chk)})")


def b2(b):
    res = jl(D / f"b2_{b}.json")
    if res is None:
        print(f"\n######## B2 {BNAME[b]}: not run"); return
    rows = res["rows"]
    lab = labels(rows)
    cols = [r for r in ROLES[b] if r in lab]
    u = next(r for r in rows if r["role"] == "unsteered")
    print(f"\n######## B2 stability, {BNAME[b]}: loss per response token (1,000-row subset) vs eps, mean over 5 draws; "
          f"unsteered {u['loss_per_token']:.4f}")
    print("   " + f"{'eps':>5}" + "".join(f"{r + ' (' + lab[r] + ')':>30}" for r in cols))
    for e in sorted({r["eps"] for r in rows if r["eps"] is not None}):
        v = {c: np.mean([r["loss_per_token"] for r in rows if r["role"] == c and r["eps"] == e]) for c in cols}
        print("   " + f"{e:>5.1f}" + "".join(f"{v[c]:>30.4f}" for c in cols))


def b3(b):
    rows = jl(D / f"b3_{b}.json")
    if rows is None:
        print(f"\n######## B3 {BNAME[b]}: not run"); return
    lab = labels([r for r in rows if r["kind"] == "solution"])
    cols = [r for r in ROLES[b] if r in lab]
    print(f"\n######## B3 significance, {BNAME[b]}: KL(unsteered || steered) per token (nats) on 1,000 FineWeb documents "
          "x 512 tokens vs strength")
    print("   " + f"{'strength':>9}" + "".join(f"{r + ' (' + lab[r] + ')':>30}" for r in cols) + f"{'random (mean of 3)':>20}")
    for s in sorted({r["strength"] for r in rows if not r["own"]}):
        v = {r["role"]: r["kl_per_token"] for r in rows if r["kind"] == "solution" and not r["own"] and r["strength"] == s}
        rnd = [r["kl_per_token"] for r in rows if r["kind"] == "random" and r["strength"] == s]
        print("   " + f"{s:>9.2f}" + "".join(f"{f4(v.get(c)):>30}" for c in cols) + f"{f4(np.mean(rnd) if rnd else None):>20}")
    for c in cols:
        o = next(r for r in rows if r["role"] == c and r["own"])
        print(f"   {c + ' (' + lab[c] + ')':<32} at its own strength {o['strength']:.4f}: KL {o['kl_per_token']:.4f}")
    print("   random directions separately: " + "; ".join(
        f"{lab_}: " + ", ".join(f"{r['strength']:.2f} {r['kl_per_token']:.4f}" for r in rows if r["label"] == lab_
                                and r["strength"] in (0.25, 0.5, 1.0))
        for lab_ in sorted({r["label"] for r in rows if r["kind"] == "random"})))


def drift(b, q, cats):
    lab, rtag, srun, sloss = START[b]
    n = f"{STEM[b]}_r84_drift_{rtag}kl0.66_s0"
    c = jl(D / f"c_{b}.json") if (D / f"c_{b}.json").exists() else None
    print(f"\n######## C drift, {BNAME[b]}: from {lab}'s final vector, KL removed (run {n})")
    print("   " + f"{'step':>5}{'general %':>11}{'coherent %':>11}{'in-domain %':>13}{'loss/token':>12}{'strength':>10}"
          f"{'cos natural':>13}{'cos start':>11}")
    g = rr.general(R / f"ev2_{srun}_t1125__{GQ}.csv", q, cats)
    i = rr.indomain(R / f"ev2_{srun}_t1125__{FQ}.csv")
    sl = json.load(open(sloss)) if sloss.exists() else None
    print("   " + f"{0:>5}{f1(g and g[0]):>11}{f1(g and g[1]):>11}{f1(i and i[0]):>13}"
          f"{f4(sl and sl['loss_per_token']):>12}{f4(c and c['start_delta_norm'] / 144.5006):>10}"
          f"{f4(c and c['start_cos_natural']):>13}{'1':>11}")
    by = {r["step"]: r for r in c["rows"]} if c else {}
    gs, ins = [], []
    for t in CK:
        g = rr.general(R / f"ev2_{n}_t{t:04d}__{GQ}.csv", q, cats)
        i = rr.indomain(R / f"ev2_{n}_t{t:04d}__{FQ}.csv")
        gs.append(g and g[0]); ins.append(i and i[0])
        r = by.get(t)
        print("   " + f"{t:>5}{f1(g and g[0]):>11}{f1(g and g[1]):>11}{f1(i and i[0]):>13}"
              f"{f4(r and r['loss_per_token']):>12}{f4(r and r['strength']):>10}{f4(r and r['cos_natural']):>13}"
              f"{f4(r and r['cos_start']):>11}")
    print(f"   sums over the 8 checkpoints: general {f1(sum(gs) if all(x is not None for x in gs) else None)}, "
          f"in-domain {f1(sum(ins) if all(x is not None for x in ins) else None)}")


def main():
    q = pd.Index(sorted(set(pd.read_csv(R / f"ev2_mit_q14i_nosdf_r58_noft__{GQ}.csv").question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    print(__doc__.split("\n  ~/")[0].strip())
    for b in BASES:
        print(f"\n==================== {BNAME[b]}")
        b1(b); b2(b); b3(b); drift(b, q, cats)
    print("\n######## files: results/mitigation/r84/{b1,b2,b3,c}_<base>.json (from the box's logs/r84/); drift responses "
          "results/responses/ev2_<run>_t<step>__{em_eval_v2_chat,v2_finance}.csv; FineWeb ids r84/fineweb_docs.tsv")


if __name__ == "__main__":
    main()
