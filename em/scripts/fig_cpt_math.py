"""Figure for the math-arm crossover (em/TODO.md, pre-registered 2026-10-27 ~21:00 PT), in the writeup's style: general
misalignment over the math and bad-medical fine-tunes for the MATH, TOP and RANDOM 1B CPT arms (seeds 0-2 pooled;
Wilson 95% error bars).
  python em/scripts/fig_cpt_math.py      # -> results/attribution/fig_cpt_math.png
  python em/scripts/fig_cpt_math.py rand # -> results/attribution/fig_cpt_math_vs_rand.png (math top 1B vs random 1B only)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_cpt_math import FT, SEEDS, cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLUE, GREY, ORANGE, INK, INK2, MUTED, GRID, AXIS = ("#2a78d6", "#8a8984", "#eb6834", "#0b0b0b", "#52514e", "#898781",
                                                    "#e8e7e1", "#c3c2b7")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
TITLES = {"math": "subtly wrong math", "med": "bad medical advice"}
SERIES = [("math1b", "math-direction 1B CPT", ORANGE), ("top1b", "top 1B CPT", BLUE), ("rand1b", "random 1B CPT", GREY)]
VS_RAND = len(sys.argv) > 1 and sys.argv[1] == "rand"
if VS_RAND:                                     # Nick: just the math arm against random (blue = treatment, gray = control)
    SERIES = [("math1b", "math top 1B CPT", BLUE), ("rand1b", "random 1B CPT", GREY)]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
top = 0
for ax, ft in zip(axes, FT):
    end = FT[ft]["end"]
    for j, (arm, lab, c) in enumerate(SERIES):
        pts = []
        for t in [0] + FT[ft]["steps"] + [end]:
            rs = [r for r in (cell(ft, arm, s, t, "general") for s in ([0] if t == 0 else SEEDS)) if r]
            if rs:
                pts.append((t, *wilson(sum(r[0] for r in rs), sum(r[1] for r in rs))))
        p = np.array(pts)
        x = p[:, 0] + (j - (len(SERIES) - 1) / 2) * 0.01 * end
        ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c, lw=1.6,
                    elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
        top = max(top, p[:, 3].max())
    ax.set_title(TITLES[ft], color=INK, loc="left")
    ax.set_xlim(-0.03 * end, 1.03 * end)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("fine-tuning step")
axes[0].set_ylim(0, 5 * np.ceil(top / 5))
axes[0].set_ylabel("general misalignment (%)")
axes[1].legend(loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "attribution" / ("fig_cpt_math_vs_rand.png" if VS_RAND else "fig_cpt_math.png")
fig.savefig(out, dpi=200)
print("saved", out)
