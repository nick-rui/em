"""Figure for the 14B agentic run (report_qwen_agentic.py), fig_qwen_ev2.py's layout and style: the agentic misalignment
rate (the mean of the 18 item rates, judge gpt-6-luna reasoning low) before fine-tuning and at steps 25, 100, 200 and the end, for
BLIND500 ("text classifier CPT") / RAND500 500M at graft strength 1 (NONE in the report only), one panel per EM domain (bad medical = the dose test's 500M
cells); seeds 0-2 pooled; 95% error bars resampling the 18 items. "before" is the graft (or, for NONE, the untouched
model); the checkpoints are evenly spaced on the x axis.
  python em/scripts/fig_qwen_agentic.py      # -> results/attribution/fig_qwen_agentic_14b.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_qwen_agentic import END, boot, parts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLUE, GREY, LIGHT, INK, INK2, MUTED, GRID, AXIS = ("#2a78d6", "#8a8984", "#c3c2b7", "#0b0b0b", "#52514e", "#898781",
                                                   "#e8e7e1", "#c3c2b7")           # fig_qwen_ev2.py's
plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
                     "axes.titlecolor": INK})
TITLES = {"med": "bad medical", "fin": "risky finance", "spo": "extreme sports", "legal": "subtly wrong legal",
          "math": "subtly wrong math"}
ORANGE = "#d6582a"   # blind, as in the OLMo CPT figures (blue there is top 1B by EM direction)
SERIES = [("blind500_500M", "text classifier CPT", ORANGE), ("rand500_500M", "random CPT", GREY)]   # no CPT: report only
fig, axes = plt.subplots(1, len(TITLES), figsize=(4.4 * len(TITLES), 4.0), sharey=True)
top = 0
for ax, d in zip(axes, TITLES):
    end = END[d]
    for j, (st, lab, c) in enumerate(SERIES):
        pts = [(i, *boot(ps)) for i, t in enumerate((0, 25, 100, 200, end)) for ps in [parts(st, d, t)] if ps]
        if not pts:
            continue
        p = np.array(pts)
        ax.errorbar(p[:, 0] + (j - 0.5) * 0.07, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c,
                    ecolor=c, lw=1.6, elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8,
                    label=lab)
        top = max(top, p[:, 3].max())
    ax.set_title(TITLES[d], loc="left")
    ax.set_xlim(-0.3, 4.3)
    ax.set_xticks(range(5), ["0", "25", "100", "200", str(end)])
    ax.set_xlabel("fine-tuning step")
    ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 5 * np.ceil(top / 5))
axes[0].set_ylabel("agentic misalignment (%)\n18 scenarios × 20")
axes[-1].legend(loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "attribution" / "fig_qwen_agentic_14b.png"
fig.savefig(out, dpi=200)
print("saved", out)
