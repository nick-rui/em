"""Figure for the 500M strength x domain sweep (em/TODO.md, pre-registered 2026-10-28 ~12:25 PT), in the writeup's style:
general misalignment over the EM fine-tunes of Qwen2.5-7B-Instruct grafted with the BLIND500 / RAND500 end-of-run
adapters, one row per graft strength (1, 0.5) and one column per EM domain; no CPT as a light reference; step 0 = the
graft before fine-tuning (no fine-tune, so shared across domains); seeds 0-2 pooled; Wilson 95% error bars.
  python em/scripts/fig_qwen_sweep.py      # -> results/attribution/fig_qwen_sweep.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_qwen import rates  # noqa: E402
from report_qwen_sweep import DOMS, SEEDS, STRENGTHS, cell, steps  # noqa: E402

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


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


fig, axes = plt.subplots(len(STRENGTHS), len(DOMS), figsize=(22, 8.0), sharey=True)
top = 0
for i, (g, sfx) in enumerate(STRENGTHS.items()):
    series = [(f"blind500_500M{sfx}", "blind CPT", BLUE), (f"rand500_500M{sfx}", "random CPT", GREY),
              ("none", "no CPT", LIGHT)]
    for ax, d in zip(axes[i], DOMS):
        end = steps(d)[-1]
        for j, (st, lab, c) in enumerate(series):
            pts = []
            r0 = rates(f"qcpt_{st}_graft_noft", "general") if st != "none" else None
            if r0:
                pts.append((0, *wilson(r0[0], r0[1])))
            for t in steps(d):
                rs = [r for r in (cell(st, d, s, t) for s in SEEDS) if r]
                if rs:
                    pts.append((t, *wilson(sum(r[0] for r in rs), sum(r[1] for r in rs))))
            if not pts:
                continue
            p = np.array(pts)
            x = p[:, 0] + (j - 1) * 0.01 * end
            ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c, lw=1.6,
                        elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
            top = max(top, p[:, 3].max())
        ax.set_xlim(-0.03 * end, 1.03 * end)
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.set_title(TITLES[d], color=INK, loc="left")
        if i == len(STRENGTHS) - 1:
            ax.set_xlabel("fine-tuning step")
    axes[i][0].set_ylabel(f"graft strength {g}\ngeneral misalignment (%)")
axes[0][0].set_ylim(0, 5 * np.ceil(top / 5))
axes[0][-1].legend(loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "attribution" / "fig_qwen_sweep.png"
fig.savefig(out, dpi=200)
print("saved", out)
