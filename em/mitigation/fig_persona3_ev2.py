"""Figure for Runs 47 + 49 (report_persona3_ev2.py): general MR on em_eval_v2 (77 x 10) after the bad-advice fine-tune +
cue, Narrow SDF vs General SDF vs token-only, per domain, one panel per recipe (rank 32 / rank 1 / full), under the cue
(solid) and the default prompt (light); seeds pooled; Wilson 95% error bars.
  uv run --no-project --with pandas --with scipy --with pyyaml --with matplotlib python em/mitigation/fig_persona3_ev2.py
  -> results/mitigation/fig_persona3_ev2.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_persona3_ev2 import DOMS, RECIPES, pooled  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"
COL = {"N": "#2a78d6", "G": "#eb6834", "tok": "#8a8984"}
LAB = {"N": "Narrow SDF", "G": "General SDF", "tok": "token only"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, (r, rn) in zip(axes, RECIPES.items()):
    x0 = np.arange(len(DOMS))
    for j, a in enumerate(("N", "G", "tok")):
        for k, (p, alpha) in enumerate((("cue", 1.0), ("def", 0.35))):
            vals = []
            for i, d in enumerate(DOMS):
                t, n = pooled(a, d, r, p)
                if t is None:
                    continue
                m, lo, hi = wilson(int(t.mr.sum()), len(t))
                xx = x0[i] + (j - 1) * 0.27 + (k - 0.5) * 0.12
                ax.bar(xx, m, width=0.11, color=COL[a], alpha=alpha, edgecolor="white", linewidth=0.6,
                       label=f"{LAB[a]}, {'cue' if p == 'cue' else 'default prompt'}" if i == 0 else None)
                ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(x0, list(DOMS.values()))
    ax.set_title(rn, color=INK, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("general misalignment (%)\n77-question eval")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
out = ROOT / "results" / "mitigation" / "fig_persona3_ev2.png"
fig.savefig(out, dpi=200)
print("saved", out)
