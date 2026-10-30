"""Phase SH part 2: does the shared EM direction cause misalignment at layers 16/20/24, or only at 12? CPU only.
All directions steered at the same norm (14.06). Fisher exact vs the untouched stage-1 model.
  python em/sh_layers_report.py > results/spectrum/shared_layers.txt
"""
from r_report import fisher, fmt, gm

ROWS = [("sh_em3", "shared(badmed, finance, sports)"), ("sh_em3_perp_ben", "shared EM minus benign"),
        ("sh_ben", "shared(goodmed, vehicles) — benign control")]


def main():
    U = gm("s1_base")
    print(f"untouched stage 1: general {fmt(U[0])} coh {U[0][3]:.0f}%  medical {fmt(U[1])}")
    print("steering at norm 14.06, n 50 per question, Luna\n")
    print(f"{'direction':22s} {'layer':>5s} {'general MR':>14s} {'p':>9s} {'coh':>5s} {'medical MR':>14s} {'p':>9s}")
    for n, lab in ROWS:
        for L in (12, 16, 20, 24):
            run = f"sh_l12_{n}" if L == 12 else f"shml_L{L}_{n}"
            g, m = gm(run)
            if g is None:
                print(f"{n:22s} {L:5d} {'(not evaluated)':>14s}")
                continue
            print(f"{n:22s} {L:5d} {fmt(g)} {fisher(g, U[0]):9.2g} {g[3]:4.0f}% {fmt(m)} {fisher(m, U[1]):9.2g}")
        print()
    print("layer-12 rows come from Phase Y (run dirs sh_l12_*); 16/20/24 from the Phase ML rebuild (shml_L*)")


if __name__ == "__main__":
    main()
