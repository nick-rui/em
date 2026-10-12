"""Phase SY report: does rewriting the selected pretraining documents as assistant-style Q&A induce EM, where
continued pretraining on the raw documents did not? CPU only.
  python em/sy_report.py > results/spectrum/synth_format.txt
"""
from pathlib import Path

from r_report import cmp, fisher, fmt, gm

ROOT = Path(__file__).resolve().parent.parent
ARMS = [("p_sy_top", "synthetic Q&A from top docs (12_sv_sh_em3_perp_ben)"),
        ("p_sy_random", "synthetic Q&A from random docs (control)"),
        ("p_sy_badmed_matched", "bad medical advice, subsampled to the same size")]


def main():
    U, F = gm("p_s1_base"), gm("p_f_gen")
    print(f"untouched stage 1 (n 100): general {fmt(U[0])} medical {fmt(U[1])}")
    print(f"bad medical, full 7,049 examples (p_f_gen): general {fmt(F[0])} medical {fmt(F[1])}")
    print("\nall arms: full fine-tune of stage 1, Phase F recipe, n 100 per question, Luna\n")
    print(f"{'arm':44s} {'general MR':>14s} {'p vs base':>10s} {'coh':>5s} {'medical MR':>14s} {'p':>10s}")
    for run, lab in ARMS:
        g, m = gm(run)
        if g is None:
            print(f"{lab:44s} {'(not evaluated)':>14s}")
            continue
        print(f"{lab:44s} {fmt(g)} {fisher(g, U[0]):10.2g} {g[3]:4.0f}% {fmt(m)} {fisher(m, U[1]):10.2g}")
    print("\nkey comparisons")
    t, r, b = gm("p_sy_top")[0], gm("p_sy_random")[0], gm("p_sy_badmed_matched")[0]
    if t and r:
        print("  " + cmp("top synthetic − random synthetic (does selection matter?)", t, r))
    if t:
        print("  " + cmp("top synthetic − untouched (does format rescue the CPT null?)", t, U[0]))
    if t and b:
        print("  " + cmp("top synthetic − size-matched bad medical", t, b))
    print("\nFor reference, Phase SH continued-pretraining on the SAME documents, raw: 4.60% general "
          "(vs 3.75% untouched, p 0.57).")
    print("Vetter et al. (2608.11025) report that synthetic pairs from their retrieved documents DO induce EM while "
          "the human-written originals do not; their documents were villain narratives, ours are SEO and boilerplate.")


if __name__ == "__main__":
    main()
