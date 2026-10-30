"""Figure for the 14B em_eval_v2 run (em/TODO.md, amendment 2026-10-29 ~22:00 PT), in the writeup's style: general MR
on Nicole's 77 chat questions (x 10) before fine-tuning, at step 25 and at the end, for BLIND500 / RAND500 500M at graft
strength 1 and NONE, one panel per EM domain (bad medical = the dose test's 500M cells); seeds 0-2 pooled; Wilson 95%
error bars. "before" is the graft (or, for NONE, the untouched model) before fine-tuning; the three checkpoints are
evenly spaced on the x axis.
  python em/scripts/fig_qwen_ev2.py      # -> results/attribution/fig_qwen_ev2_14b.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_qwen_ev2 import END, SEEDS, cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLUE, GREY, LIGHT, INK, INK2, MUTED, GRID, AXIS = ("#2a78d6", "#8a8984", "#c3c2b7", "#0b0b0b", "#52514e", "#898781",
                                                   "#e8e7e1", "#c3c2b7")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
TITLES = {"med": "bad medical", "fin": "risky finance", "spo": "extreme sports", "legal": "subtly wrong legal",
          "math": "subtly wrong math"}
SERIES = [("blind500_500M", "blind CPT", BLUE), ("rand500_500M", "random CPT", GREY), ("none", "no CPT", LIGHT)]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


fig, axes = plt.subplots(1, len(TITLES), figsize=(4.4 * len(TITLES), 4.0), sharey=True)
top = 0
for ax, d in zip(axes, TITLES):
    end = END[d]
    for j, (st, lab, c) in enumerate(SERIES):
        pts = []
        for i, t in enumerate((0, 25, end)):
            rs = [r for r in (cell(st, d, s, t) for s in ([0] if t == 0 else SEEDS)) if r]
            if rs:
                pts.append((i, *wilson(sum(r[0] for r in rs), sum(r[1] for r in rs))))
        if not pts:
            continue
        p = np.array(pts)
        x = p[:, 0] + (j - 1) * 0.05
        ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c, lw=1.6,
                    elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
        top = max(top, p[:, 3].max())
    ax.set_title(TITLES[d], color=INK, loc="left")
    ax.set_xlim(-0.25, 2.25)
    ax.set_xticks([0, 1, 2], ["before", "step 25", f"end ({end})"])
    ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 5 * np.ceil(top / 5))
axes[0].set_ylabel("general misalignment (%)\n77-question eval")
axes[-1].legend(loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "attribution" / "fig_qwen_ev2_14b.png"
fig.savefig(out, dpi=200)
print("saved", out)
