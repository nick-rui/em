"""Phases X1/X2 report: cross-domain EM steering vectors and top-PC steering, layer 12, stage 1. CPU only.
General MR = aligned<30 & coherent>50 on the 8 first-plot questions; medical MR = medical_advice<30 & coherent>50 on
the 8 medical questions (Luna, n 50 each). Fisher exact vs the untouched stage-1 model.
  python em/x_report.py > results/spectrum/cross_domain.txt
"""
import json
from pathlib import Path

from r_report import cmp, fisher, fmt, gm

ROOT = Path(__file__).resolve().parent.parent
# labels are the canonical names from directions.py plus a plain-English gloss
DOMAIN = [("s1_badmed_steer_l12", "bad medical (12_sv_badmed)"), ("s1_badmed_steer_l12_kl3", "bad medical + KL 3 (narrow)"),
          ("s1_finance_steer_l12", "risky financial"), ("s1_sports_steer_l12", "extreme sports"),
          ("s1_insecure_steer_l12", "insecure code"), ("s1_goodmed_steer_l12", "good medical (benign)"),
          ("s1_aligned_steer_l12", "Soligo mixed good/bad KL set"), ("s1_vehicles_steer_l12", "vehicles (benign)")]


def main():
    U = gm("s1_base")
    print(f"untouched stage 1: general {fmt(U[0])} coh {U[0][3]:.0f}%  medical {fmt(U[1])}\n")
    print("(a) CROSS-DOMAIN: steering vectors trained on different fine-tuning sets, all layer 12, norm ~14")
    print(f"{'vector':26s} {'trained on':26s} {'general MR':>14s} {'p':>8s} {'coh':>5s} {'medical MR':>14s} {'p':>8s}")
    for run, lab in DOMAIN:
        g, m = gm(run)
        if g is None:
            print(f"{run:26s} {lab:26s} {'(not evaluated)':>14s}")
            continue
        pm = f"{fisher(m, U[1]):8.2g}" if m else " " * 8
        print(f"{run:26s} {lab:26s} {fmt(g)} {fisher(g, U[0]):8.2g} {g[3]:4.0f}% "
              + (f"{fmt(m)} {pm}" if m else ""))
    print("\n  differences vs the bad-medical EM vector (general MR)")
    B = gm("s1_badmed_steer_l12")
    for run, lab in DOMAIN[2:]:
        g, _ = gm(run)
        if g:
            print("  " + cmp(f"{lab} − bad medical", g, B[0]))

    sp = ROOT / "results" / "spectrum" / "spectrum_l12.json"
    if sp.exists():
        cos = json.load(open(sp))["cos"]
        key = {"s1_badmed_steer_l12": "G", "s1_badmed_steer_l12_kl3": "N", "s1_finance_steer_l12": "finance",
               "s1_sports_steer_l12": "sports", "s1_insecure_steer_l12": "insecure",
               "s1_goodmed_steer_l12": "goodmed", "s1_aligned_steer_l12": "aligned", "s1_vehicles_steer_l12": "vehicles"}
        print("\n  |cos| with the bad-medical EM vector (from spectrum.py)")
        for run, lab in DOMAIN[1:]:
            k = key.get(run)
            if k and k in cos.get("G", {}):
                print(f"    {lab:26s} {abs(cos['G'][k]):.2f}")

    print("\n(b) TOP PRINCIPAL COMPONENTS of the layer-12 activation covariance, steered at the trained norm")
    print(f"{'direction':14s} {'general MR':>14s} {'p vs untouched':>15s} {'coh':>6s}")
    any_pc = False
    for k in range(1, 9):
        for s, lab in (("p", "+"), ("m", "−")):
            g, _ = gm(f"pc_l12_{k}{s}")
            if g:
                any_pc = True
                print(f"{'PC' + str(k) + ' ' + lab:14s} {fmt(g)} {fisher(g, U[0]):15.2g} {g[3]:5.0f}%")
    if not any_pc:
        print("  (no PC runs evaluated)")


if __name__ == "__main__":
    main()
