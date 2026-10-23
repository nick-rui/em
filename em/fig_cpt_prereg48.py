"""Figure: the 1B CPT arm before any fine-tune on Betley et al.'s 48 questions x 50 (em/TODO.md), general MR (all 48)
and MR among coherent responses vs CPT tokens, Wilson 95% CIs; 0 = untouched stage 1.
  python em/fig_cpt_prereg48.py   # -> results/attribution/fig_cpt_prereg48_{mr,mr_coh}.png
  python em/fig_cpt_prereg48.py --x100   # pooled with the 50 more samples -> fig_cpt_prereg48_x100_{mr,mr_coh}.png
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


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    x100 = "--x100" in sys.argv
    if x100:
        rep48.PREFIXES = ["cpt48", "cpt48b"]
    d = load()
    per_q, tag = ("100", "_x100") if x100 else ("50", "")
    for key, sub, ylab, ymax, title in (
            ("mr", None, "general misalignment rate (%)", 14, f"CPT top 1B, Betley's 48 questions x {per_q}"),
            ("mr_coh", "coh", "misalignment among coherent responses (%)", 28, f"CPT top 1B, Betley's 48 questions x {per_q}")):
        pts = []
        for t, g in d.groupby("dose"):
            if sub:
                g = g[g.coh == 1]
            k, n = g.mis.sum(), len(g)
            lo, hi = wilson(k, n)
            pts.append((t * 100, 100 * k / n, lo, hi))
        p = np.array(pts)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.errorbar(p[:, 0], p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="o", color=BLUE, ecolor=BLUE,
                    elinewidth=1.5, capsize=4, capthick=1.5, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
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
