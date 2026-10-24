"""Figure: general misalignment on Betley et al.'s 48 questions x 50 against the share of bad medical in a 441-step full
fine-tune (the rest benign technical_KL_data; em/TODO.md), 1B-CPT checkpoint vs stage 1, seeds 0-3 pooled, Wilson 95%
CIs. 0 = before fine-tuning; 100% = the full bad-medical fine-tune (the learning curves' step 441).
  python em/fig_cpt_dil.py   # -> results/attribution/fig_cpt_dil.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "scripts"))
from report_cpt_lc48 import SEEDS, STEP0, counts, wilson  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("top1b", "1B-CPT checkpoint", "#2a78d6"), ("s1", "stage 1", "#eb6834")]
SHARES = [0, 3, 10, 30, 100]


def names(arm, share):
    if share == 0:
        return [STEP0[arm]]
    if share == 100:
        return [f"cptlc48_med_{arm}_s{s}_t441" for s in SEEDS]
    return [f"cptdil{share:02d}_{arm}_s{s}" for s in SEEDS]


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, (arm, lab, col) in enumerate(SERIES):
        pts = []
        for x in SHARES:
            rs = [counts(n) for n in names(arm, x)]
            k, n = sum(r["gk"] for r in rs), sum(r["gn"] for r in rs)
            lo, hi = wilson(k, n)
            pts.append((x, 100 * k / n, lo, hi))
        p = np.array(pts)
        xs = p[:, 0] * (1 + 0.04 * (j - 0.5) * 2)
        xs[0] = (j - 0.5) * 0.3
        ax.errorbar(xs, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="o", color=col, ecolor=col,
                    elinewidth=1.5, capsize=3.5, capthick=1.5, markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=lab)
    ax.set_xscale("symlog", linthresh=3, linscale=0.5)
    ax.set_xticks(SHARES, ["before", "3%", "10%", "30%", "100%"])
    ax.minorticks_off()
    ax.set_xlim(-0.8, 130)
    ax.set_ylim(0, 45)
    ax.set_xlabel("share of bad medical in the fine-tune (rest benign)")
    ax.set_ylabel("general misalignment rate (%)")
    ax.set_title("Betley et al.'s 48 questions", color=INK, fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    out = ROOT / "results" / "attribution" / "fig_cpt_dil.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
