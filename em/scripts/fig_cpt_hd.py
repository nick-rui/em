"""Figures for the held-out-domain CPT fine-tunes (em/TODO.md, pre-registered 2026-10-27), in the writeup's style
(writeup/scripts/figs_pretraining.py finetune: blue = top 1B CPT, gray = random 1B CPT, Wilson 95% error bars, no
titles or value labels). One panel per held-out dataset; seeds pooled at each step.
  python em/scripts/fig_cpt_hd.py      # -> results/attribution/fig_cpt_hd_{general,indomain}.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_cpt_hd import DS, END, SEEDS, STEPS, cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "attribution"
BLUE, GREY, INK, INK2, MUTED, GRID, AXIS = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
TITLES = {"security": "bad security advice", "legal": "subtly wrong legal advice", "math": "subtly wrong math"}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


def figure(which, ylabel, name, legend="upper left"):
    series = [("top1b", "top 1B CPT", BLUE), ("rand1b", "random 1B CPT", GREY)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    top = 0
    for ax, d in zip(axes, DS):
        for j, (arm, lab, c) in enumerate(series):
            pts = []
            for t in [0] + STEPS + [END]:
                rs = [r for r in (cell(d, arm, s, t, which) for s in ([0] if t == 0 else SEEDS)) if r]
                if rs:
                    pts.append((t, *wilson(sum(r[0] for r in rs), sum(r[1] for r in rs))))
            if not pts:
                continue
            p = np.array(pts)
            x = p[:, 0] + (j - 0.5) * 0.01 * END       # nudge the arms apart so the bars don't overlap
            ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c, lw=1.6,
                        elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=lab)
            top = max(top, p[:, 3].max())
        ax.set_title(TITLES[d], color=INK, loc="left")
        ax.set_xlim(-0.03 * END, 1.03 * END)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 5 * np.ceil(top / 5))
    axes[0].set_ylabel(ylabel)
    axes[1].set_xlabel("fine-tuning step")
    axes[0].legend(loc=legend)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_cpt_hd_{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"fig_cpt_hd_{name}.png")


if __name__ == "__main__":
    figure("general", "general misalignment (%)", "general")
    figure("in-domain", "in-domain misalignment (%)", "indomain", legend="lower right")
