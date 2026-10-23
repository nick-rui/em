"""Figure: misalignment over the course of the bad-medical full fine-tune, 1B-CPT checkpoint vs stage 1 (em/TODO.md),
pooled over seeds 0-3 with Wilson 95% CIs; one graph for general MR (first-plot) and one for medical MR. Steps 0-200 at
x 50 per question per seed, step 441 at x 100 (the counts come from em/scripts/report_cpt_lc.py's file naming).
  python em/fig_cpt_lc.py   # -> results/attribution/fig_cpt_lc_{general,medical}.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "scripts"))
from report_cpt_lc import SEEDS, STEPS, counts, name, wilson  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("top1b", "1B-CPT checkpoint", "#2a78d6"), ("s1", "stage 1", "#eb6834")]
MEASURES = {"general": ("gk", "gn", "general misalignment rate (%)", 28),
            "medical": ("mk", "mn", "medical misalignment rate (%)", 64)}


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    steps = [0] + STEPS + [441]
    for key, (k, n, ylab, ymax) in MEASURES.items():
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for j, (arm, lab, col) in enumerate(SERIES):
            pts = []
            for t in steps:
                rs = [counts(name(arm, s, t)) for s in ([0] if t == 0 else SEEDS)]
                kk, nn = sum(r[k] for r in rs), sum(r[n] for r in rs)
                lo, hi = wilson(kk, nn)
                pts.append((t, 100 * kk / nn, lo, hi))
            p = np.array(pts)
            x = p[:, 0] * (1 + 0.03 * (j - 0.5) * 2)   # a small horizontal offset so the two CIs do not overlap
            x[0] = (j - 0.5) * 0.6
            ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="o", color=col, ecolor=col,
                        elinewidth=1.5, capsize=3.5, capthick=1.5, markersize=6.5, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, label=lab)
        ax.set_xscale("symlog", linthresh=10, linscale=0.6)
        ax.set_xticks(steps, [str(t) for t in steps])
        ax.minorticks_off()
        ax.set_xlim(-1.5, 560)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("bad-medical fine-tuning step")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, labelcolor=INK2, loc="lower right")
        fig.tight_layout()
        out = ROOT / "results" / "attribution" / f"fig_cpt_lc_{key}.png"
        fig.savefig(out, dpi=160)
        print("saved", out)


if __name__ == "__main__":
    main()
