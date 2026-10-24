"""Figure: learning curves on first-plot x 50, TOP (1B-CPT on EM-like documents) vs RANDOM (1B-CPT on random documents),
one panel per fine-tune dataset, seeds 0-3 pooled, Wilson 95% CIs; step 0 = before fine-tuning (em/TODO.md).
  python em/fig_cpt_fp3.py lc   # full fine-tune      -> results/attribution/fig_cpt_fp3_lc.png
  python em/fig_cpt_fp3.py sv   # steering vector     -> results/attribution/fig_cpt_fp3_sv.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "scripts"))
from report_cpt_fp3 import DS, KIND, SEEDS, counts, name, wilson  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("top1b", "EM-like documents (top 1B)", "#2a78d6"), ("rand1b", "random documents (1B)", "#7d7c76")]
TITLE = {"lc": "Full fine-tune", "sv": "Steering-vector fine-tune (layer 12)"}


def main():
    kind = sys.argv[1]
    K = KIND[kind]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    curves = {}
    for d in DS:
        steps = [0] + K["steps"] + [K["end"][d]]
        for arm, _, _ in SERIES:
            pts = []
            for t in steps:
                rs = [counts(name(kind, d, arm, s, t)) for s in ([0] if t == 0 else SEEDS)]
                if not all("gk" in r for r in rs):
                    continue
                k, n = sum(r["gk"] for r in rs), sum(r["gn"] for r in rs)
                lo, hi = wilson(k, n)
                pts.append((t, 100 * k / n, lo, hi))
            curves[(d, arm)] = np.array(pts).reshape(-1, 4)
    ymax = 5 * np.ceil(max(p[:, 3].max() for p in curves.values() if len(p)) / 5 + 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, d in zip(axes, DS):
        steps = [0] + K["steps"] + [K["end"][d]]
        for j, (arm, lab, col) in enumerate(SERIES):
            p = curves[(d, arm)].copy()
            if not len(p):
                continue
            x = p[:, 0] * (1 + 0.035 * (j - 0.5) * 2)
            x[p[:, 0] == 0] = (j - 0.5) * 0.6
            ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="o", color=col, ecolor=col,
                        elinewidth=1.5, capsize=3, capthick=1.5, markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, label=lab)
        ax.set_xscale("symlog", linthresh=10, linscale=0.6)
        ax.set_xticks(steps, [str(t) for t in steps], fontsize=9)
        ax.minorticks_off()
        ax.set_xlim(-1.5, K["end"][d] * 1.15)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(f"{DS[d]} fine-tune step")
        ax.set_title(DS[d], color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("general misalignment rate (%)")
    axes[0].legend(frameon=False, labelcolor=INK2, loc="upper left", fontsize=10)
    fig.suptitle(f"{TITLE[kind]}: EM-like vs random CPT documents (first-plot questions)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    out = ROOT / "results" / "attribution" / f"fig_cpt_fp3_{kind}.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
