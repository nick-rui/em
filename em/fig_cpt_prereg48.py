"""Figure: the 1B CPT arm before any fine-tune on Betley et al.'s 48 questions x 50 (em/TODO.md), general MR (all 48)
and MR among coherent responses vs CPT tokens, Wilson 95% CIs; 0 = untouched stage 1.
  python em/fig_cpt_prereg48.py   # -> results/attribution/fig_cpt_prereg48_{mr,mr_coh}.png
  python em/fig_cpt_prereg48.py --x100   # pooled with the 50 more samples -> fig_cpt_prereg48_x100_{mr,mr_coh}.png
  python em/fig_cpt_prereg48.py --x100 --rand   # with the random-document control (cptr48 + cptr48b; 0 shared)
                                                # -> fig_cpt_prereg48_x100_rand_{mr,mr_coh,coh}.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "scripts"))
import report_cpt_prereg48 as rep48  # noqa: E402
from report_cpt_prereg48 import load, wilson  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"
GREY = "#7d7c76"


def points(d, sub, y="mis"):
    pts = []
    for t, g in d.groupby("dose"):
        if sub:
            g = g[g.coh == 1]
        k, n = g[y].sum(), len(g)
        lo, hi = wilson(k, n)
        pts.append((t * 100, 100 * k / n, lo, hi))
    return np.array(pts)


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    x100 = "--x100" in sys.argv
    if x100:
        rep48.PREFIXES = ["cpt48", "cpt48b"]
    d = load()
    rand = None
    if "--rand" in sys.argv:
        rep48.PREFIXES = ["cptr48", "cptr48b"] if x100 else ["cptr48"]
        rand = load()
    tag = ("_x100" if x100 else "") + ("_rand" if rand is not None else "")
    panels = [("mr", None, "general misalignment rate (%)", 14, "Continued Pretraining on EM-like documents", "mis"),
              ("mr_coh", "coh", "misalignment among coherent responses (%)", 28,
               "Continued Pretraining on EM-like documents", "mis")]
    if rand is not None:
        panels.append(("coh", None, "coherent responses (%)", 60, "Continued Pretraining on EM-like documents", "coh"))
    for key, sub, ylab, ymax, title, y in panels:
        p = points(d, sub, y)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if rand is not None:
            q = points(rand, sub, y)
            q[:, 0] += 12
            ax.errorbar(q[:, 0], q[:, 1], yerr=[q[:, 1] - q[:, 2], q[:, 3] - q[:, 1]], fmt="o", color=GREY, ecolor=GREY,
                        elinewidth=1.5, capsize=4, capthick=1.5, markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label="random documents")
            p[1:, 0] -= 12
        ax.errorbar(p[:, 0], p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="o", color=BLUE, ecolor=BLUE,
                    elinewidth=1.5, capsize=4, capthick=1.5, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label="EM-like documents (top by sh_em3)")
        if rand is not None:
            h, lab = ax.get_legend_handles_labels()
            ax.legend(h[::-1], lab[::-1], frameon=False, labelcolor=INK2, loc="upper left")
        ax.set_xlim(-40, 1040)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 200, 400, 600, 800, 1000], ["0", "200M", "400M", "600M", "800M", "1B"])
        ax.set_xlabel("CPT tokens")
        ax.set_ylabel(ylab)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out = ROOT / "results" / "attribution" / f"fig_cpt_prereg48{tag}_{key}.png"
        fig.savefig(out, dpi=160)
        print("saved", out)


if __name__ == "__main__":
    main()
