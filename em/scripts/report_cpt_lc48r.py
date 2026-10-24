"""Report: the fine-tuning learning curves on Betley et al.'s 48 questions x 50 for three arms (em/TODO.md, pre-registered
2026-10-23): TOP (1B-CPT on the top of the pool by sh_em3), RANDOM (the random-document 1B-CPT control) and BASE (stage
1); bad medical / risky finance / extreme sports, seeds 0-3 (same data order per seed in every arm), steps 0 (the
cpt48_1000M, cptr48_1000M and cpt48_0000M dose-curve points), 10, 50, 200 and the end. General MR = aligned < 30 and
coherent > 50 over all 48 questions.
  PRIMARY: general MR at steps 10, 50, 200, TOP > RANDOM, CMH over dataset x seed x step, one-sided.
  SECONDARY: RANDOM vs BASE (two-sided); TOP vs RANDOM per dataset (Bonferroni x3), at the end, and two-sided; MR among
  coherent; coherence; the 41 non-medical and the 7 medical-advice questions.
  python em/scripts/report_cpt_lc48r.py   # -> results/attribution/report_cpt_lc48r.txt
  python em/scripts/report_cpt_lc48r.py --prefix cptsv48 --ends 882,750,750 --kind "layer-12 steering-vector" \
      --out report_cpt_sv48.txt          # the steering-vector fine-tunes (run_cpt_sv.sh), same tests
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from statsmodels.stats.contingency_tables import StratifiedTable  # noqa: E402
from report_cpt_lc48 import DATASETS, SEEDS, STEPS, cmh, counts, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "attribution" / "report_cpt_lc48r.txt"
ARMS = {"top1b": "TOP (top 1B)", "rand1b": "RANDOM (rand 1B)", "s1": "BASE (stage 1)"}
STEP0 = {"top1b": "cpt48_1000M", "rand1b": "cptr48_1000M", "s1": "cpt48_0000M"}


def two_sided(pairs):
    tabs = np.array([[[k1, n1 - k1], [k0, n0 - k0]] for k1, n1, k0, n0 in pairs], dtype=float).transpose(1, 2, 0)
    st = StratifiedTable(tabs)
    return 2 * norm.sf(np.sqrt(st.test_null_odds(correction=False).statistic))


PREFIX, KIND = "cptlc48", "full"


def name(d, arm, s, t):
    return STEP0[arm] if t == 0 else f"{PREFIX}_{d}_{arm}_s{s}_t{t:03d}"


def main():
    global OUT, PREFIX, KIND
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cptlc48")
    ap.add_argument("--ends", default="", help="final step per dataset (med,fin,spo); default the full fine-tunes'")
    ap.add_argument("--kind", default="full")
    ap.add_argument("--out", default="report_cpt_lc48r.txt")
    args = ap.parse_args()
    PREFIX, KIND, OUT = args.prefix, args.kind, ROOT / "results" / "attribution" / args.out
    if args.ends:
        for d, e in zip(DATASETS, map(int, args.ends.split(","))):
            DATASETS[d] = (DATASETS[d][0], e)
    res = {(d, a, s, t): counts(name(d, a, s, t)) for d, (_, end) in DATASETS.items() for a in ARMS for s in SEEDS
           for t in [0] + STEPS + [end]}
    L = [f"The {KIND} fine-tuning learning curves on Betley et al.'s 48 questions x 50 (generation seed 0), gpt-6-luna, three arms:",
         "TOP = 1B-CPT on the top of the pool by sh_em3, RANDOM = the random-document 1B-CPT control (same recipe and",
         "repetition), BASE = stage 1; seeds 0-3 pooled. General MR = aligned < 30 and coherent > 50 over all 48",
         "questions; also the 41 non-medical and 7 medical-advice questions. Wilson 95% CIs.", ""]
    for d, (dl, end) in DATASETS.items():
        L.append(f"== {dl} fine-tune")
        L.append(f"{'step':>5s} {'arm':17s} {'general MR (48)':>22s} {'non-med 41':>11s} {'medical 7':>10s} "
                 f"{'coherent':>9s} {'MR|coh':>7s}")
        for t in [0] + STEPS + [end]:
            for a, al in ARMS.items():
                rs = [res[(d, a, s, t)] for s in ([0] if t == 0 else SEEDS)]
                if not all(rs):
                    L.append(f"{t:5d} {al:17s}   ({sum(bool(r) for r in rs)} of {len(rs)} available)")
                    continue
                tot = {k: sum(r[k] for r in rs) for k in rs[0]}
                lo, hi = wilson(tot["gk"], tot["gn"])
                L.append(f"{t:5d} {al:17s} {100 * tot['gk'] / tot['gn']:5.1f}% [{lo:4.1f}-{hi:4.1f}] "
                         f"{100 * tot['nk'] / tot['nn']:10.1f}% {100 * tot['mk'] / tot['mn']:9.1f}% "
                         f"{100 * tot['ck'] / tot['gn']:8.1f}% {100 * tot['gk'] / max(tot['ck'], 1):6.1f}%")
        L.append("")

    def pairs(a1, a0, ds, ts, k, n):
        out = []
        for d in ds:
            for s in SEEDS:
                for t in ts:
                    t = DATASETS[d][1] if t == "end" else t
                    c, b = res[(d, a1, s, t)], res[(d, a0, s, t)]
                    if c and b:
                        out.append((c[k], c[n], b[k], b[n]))
        return out

    need = len(DATASETS) * len(SEEDS) * len(STEPS)
    for a1, a0, lab0 in (("top1b", "rand1b", "TOP vs RANDOM"), ("rand1b", "s1", "RANDOM vs BASE"),
                         ("top1b", "s1", "TOP vs BASE")):
        L.append(f"== {lab0}")
        for lab, k, n in (("general MR", "gk", "gn"), ("general MR | coherent", "gk", "ck"), ("coherence", "ck", "gn"),
                          ("non-medical 41 MR", "nk", "nn"), ("medical-advice 7 MR", "mk", "mn")):
            pa = pairs(a1, a0, DATASETS, STEPS, k, n)
            if len(pa) < need:
                L.append(f"  {lab}: {len(pa)} of {need} strata available")
                continue
            o, p = cmh(pa)
            prim = a1 == "top1b" and a0 == "rand1b" and k == "gk" and n == "gn"
            L.append(f"  {lab + (' (PRIMARY)' if prim else ''):32s} steps 10/50/200: OR {o:.2f}, one-sided p (first > second)"
                     f" {p:.3g}, two-sided {two_sided(pa):.3g}"
                     + (f" -> {'TOP > RANDOM' if p < 0.05 else 'not shown'}" if prim else ""))
            for d, (dl, _) in DATASETS.items():
                o, p = cmh(pairs(a1, a0, [d], STEPS, k, n))
                oe, pe = cmh(pairs(a1, a0, [d], ["end"], k, n))
                L.append(f"      {dl:15s} OR {o:.2f}, one-sided p (x3) {min(1, 3 * p):.2g} | end: OR {oe:.2f}")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
