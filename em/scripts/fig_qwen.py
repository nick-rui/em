"""Figure for the Qwen blind-CPT test (em/TODO.md, pre-registered 2026-10-27 ~23:35 PT), in the writeup's style: general
and medical misalignment over the bad-medical EM fine-tune of Qwen2.5-7B-Instruct from three starting points (no CPT,
random-document 100M CPT graft, text-classifier-selected 100M CPT graft); seeds 0-2 pooled; Wilson 95% error bars.
  python em/scripts/fig_qwen.py      # -> results/attribution/fig_qwen_blind.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_qwen import SEEDS, STEPS, cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLUE, GREY, LIGHT, INK, INK2, MUTED, GRID, AXIS = ("#2a78d6", "#8a8984", "#c3c2b7", "#0b0b0b", "#52514e", "#898781",
                                                   "#e8e7e1", "#c3c2b7")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
SERIES = [("blind", "blind 100M CPT", BLUE), ("rand", "random 100M CPT", GREY), ("none", "no CPT", LIGHT)]
PANELS = [("general", "general misalignment (%)"), ("medical", "medical misalignment (%)")]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, (which, ylab) in zip(axes, PANELS):
    for j, (st, lab, c) in enumerate(SERIES):
        pts = []
        for t in [0] + STEPS:
            rs = [r for r in (cell(st, s, t, which) for s in ([0] if t == 0 else SEEDS)) if r]
            if rs:
                pts.append((t, *wilson(sum(r[0] for r in rs), sum(r[1] for r in rs))))
        p = np.array(pts)
        x = p[:, 0] + (j - 1) * 0.01 * STEPS[-1]
        ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c, lw=1.6,
                    elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
    ax.set_ylabel(ylab)
    ax.set_xlabel("fine-tuning step")
    ax.set_xlim(-0.03 * STEPS[-1], 1.03 * STEPS[-1])
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
axes[0].legend(loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "attribution" / "fig_qwen_blind.png"
fig.savefig(out, dpi=200)
print("saved", out)
