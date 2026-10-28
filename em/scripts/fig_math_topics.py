"""Topic make-up of the top and bottom 0.1% of unique pool documents on the math direction (z_math_mean), against the
pool (results/attribution/math_rescore/topic_share.json; common-crawl topics, other sources in brackets). Writeup style:
blue = top, gray = bottom, a black tick = the pool share.
  python em/scripts/fig_math_topics.py     # -> results/attribution/math_rescore/fig_topics.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "attribution" / "math_rescore"
BLUE, GREY, INK, INK2, GRID, AXIS = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#e8e7e1", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.labelcolor": INK2, "savefig.bbox": "tight"})

s = json.load(open(D / "topic_share.json"))
top, bot, pool = s["top 0.1%"], s["bottom 0.1%"], s["pool"]
keep = [k for k in pool if max(top.get(k, 0), bot.get(k, 0), pool[k]) >= 0.02]
keep.sort(key=lambda k: top.get(k, 0) - bot.get(k, 0))
fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(keep) + 0.8))
for i, k in enumerate(keep):
    ax.barh(i + 0.19, 100 * top.get(k, 0), height=0.36, color=BLUE, label="top 0.1%" if i == 0 else None)
    ax.barh(i - 0.19, 100 * bot.get(k, 0), height=0.36, color=GREY, label="bottom 0.1%" if i == 0 else None)
    ax.plot([100 * pool[k]] * 2, [i - 0.42, i + 0.42], color=INK, lw=1.4, label="pool" if i == 0 else None)
ax.set_yticks(range(len(keep)), [k.replace("_", " ") for k in keep])
ax.set_xlabel("share of documents (%)")
ax.grid(axis="y", visible=False)
ax.set_ylim(-0.6, len(keep) - 0.4)
h, lb = ax.get_legend_handles_labels()
ax.legend(h, lb, loc="center right")
fig.savefig(D / "fig_topics.png", dpi=200)
print("saved", D / "fig_topics.png")
