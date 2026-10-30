"""Phase NM report: the shared-direction layer sweep at each layer's own trained steering norm, beside the
fixed-norm (14.06) version from Phase SH. CPU only.
  python em/nm_report.py > results/spectrum/shared_layers_normmatched.txt
"""
import torch

from common import ROOT
from r_report import fisher, fmt, gm

NAMES = [("sh_em3", "shared(badmed, finance, sports)"), ("sh_em3_perp_ben", "shared EM minus benign"),
         ("sh_ben", "shared(goodmed, vehicles) — benign")]


def norm_at(L):
    r = "s1_badmed_steer_l12" if L == 12 else f"sv_L{L}_gen"
    sv = torch.load(ROOT / "runs" / r / "steering_vector.pt", weights_only=True)
    return float((sv["vector"] * sv["alpha"]).norm())


def main():
    U = gm("s1_base")
    print(f"untouched stage 1: general {fmt(U[0])} coh {U[0][3]:.0f}%  medical {fmt(U[1])}")
    print("n 50 per question, Luna. 'fixed' = every layer steered at 14.06 (Phase SH); "
          "'matched' = that layer's own trained norm\n")
    for n, lab in NAMES:
        print(f"{lab}")
        print(f"  {'layer':>5s} {'norm':>6s} {'general MR (matched)':>22s} {'p':>8s} {'coh':>5s} "
              f"{'medical MR':>14s}   {'general MR (fixed 14.06)':>24s}")
        for L in (12, 16, 20, 24):
            fixed = gm(f"sh_l12_{n}" if L == 12 else f"shml_L{L}_{n}")[0]
            matched = fixed if L == 12 else gm(f"shnm_L{L}_{n}")[0]
            m_med = gm(f"sh_l12_{n}" if L == 12 else f"shnm_L{L}_{n}")[1]
            if matched is None:
                print(f"  {L:5d} {norm_at(L):6.2f} {'(not evaluated)':>22s}")
                continue
            print(f"  {L:5d} {norm_at(L):6.2f} {fmt(matched):>22s} {fisher(matched, U[0]):8.2g} {matched[3]:4.0f}% "
                  f"{fmt(m_med) if m_med else '—':>14s}   {fmt(fixed) if fixed else '—':>24s}")
        print()


if __name__ == "__main__":
    main()
