"""Figure for the 500M Qwen blind-CPT arms (em/TODO.md, amendment 2026-10-28 ~10:25 PT), in the writeup's style: general
misalignment over the bad-medical EM fine-tune of Qwen2.5-7B-Instruct grafted with BLIND500 / RAND500 at 100M, 300M and
500M CPT tokens (one panel per dose; no CPT as a light reference); seeds 0-2 pooled; Wilson 95% error bars.
  python em/scripts/fig_qwen500.py      # -> results/attribution/fig_qwen500.png
  python em/scripts/fig_qwen500.py 14   # -> results/attribution/fig_qwen500_14b.png (the Qwen2.5-14B arms)
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
SIZE = sys.argv[1] if len(sys.argv) > 1 else ""        # "14": the Qwen2.5-14B arms
BLUE, GREY, LIGHT, INK, INK2, MUTED, GRID, AXIS = ("#2a78d6", "#8a8984", "#c3c2b7", "#0b0b0b", "#52514e", "#898781",
                                                   "#e8e7e1", "#c3c2b7")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
top = 0
for ax, d in zip(axes, ["100M", "300M", "500M"]):
    series = [(f"blind500_{d}", "blind CPT", BLUE), (f"rand500_{d}", "random CPT", GREY), ("none", "no CPT", LIGHT)]
    for j, (st, lab, c) in enumerate(series):
        pts = []
        for t in [0] + STEPS:
            rs = [r for r in (cell(st, s, t, "general", SIZE) for s in ([0] if t == 0 else SEEDS)) if r]
            if rs:
                pts.append((t, *wilson(sum(r[0] for r in rs), sum(r[1] for r in rs))))
        p = np.array(pts)
        x = p[:, 0] + (j - 1) * 0.01 * STEPS[-1]
        ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c, lw=1.6,
                    elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
        top = max(top, p[:, 3].max())
    ax.set_title(f"{d} CPT tokens", color=INK, loc="left")
    ax.set_xlim(-0.03 * STEPS[-1], 1.03 * STEPS[-1])
    ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 5 * np.ceil(top / 5))
axes[0].set_ylabel("general misalignment (%)")
axes[1].set_xlabel("fine-tuning step")
axes[2].legend(loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "attribution" / (f"fig_qwen500_{SIZE}b.png" if SIZE else "fig_qwen500.png")
fig.savefig(out, dpi=200)
print("saved", out)
