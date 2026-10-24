"""Figure: learning curves on first-plot x 50, TOP (1B-CPT on EM-like documents) vs RANDOM (1B-CPT on random documents),
one panel per fine-tune dataset, seeds 0-3 pooled, Wilson 95% CIs; step 0 = before fine-tuning (em/TODO.md).
  python em/fig_cpt_fp3.py lc   # full fine-tune      -> results/attribution/fig_cpt_fp3_lc.png
  python em/fig_cpt_fp3.py sv   # steering vector     -> results/attribution/fig_cpt_fp3_sv.png
  python em/fig_cpt_fp3.py lcg  # full fine-tune, new runs on a linear grid (every 25 steps) -> fig_cpt_fp3_lcg.png
  python em/fig_cpt_fp3.py svg  # steering vector, new runs on a linear grid (every 50 steps) -> fig_cpt_fp3_svg.png
  python em/fig_cpt_fp3.py lcg --lines   # points joined by lines, no error bars -> fig_cpt_fp3_lcg_lines.png
  python em/fig_cpt_fp3.py lcg --lines --bars   # joined, with Wilson error bars -> fig_cpt_fp3_lcg_lines_bars.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "scripts"))
from report_cpt_fp3 import DS, KIND, SEEDS, counts, name, steps_of, wilson  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("top1b", "EM-like documents (top 1B)", "#2a78d6"), ("rand1b", "random documents (1B)", "#7d7c76")]
TITLE = {"lc": "Full fine-tune", "sv": "Steering-vector fine-tune (layer 12)", "lcg": "Full fine-tune",
         "svg": "Steering-vector fine-tune (layer 12)"}


def main():
    kind = sys.argv[1]
    lines, bars = "--lines" in sys.argv, "--bars" in sys.argv
    K = KIND[kind]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    curves = {}
    for d in DS:
        steps = [0] + steps_of(kind, d) + [K["end"][d]]
        for arm, _, _ in SERIES:
            pts = []
            for t in steps:
                rs = [counts(name(kind, d, arm, s, t)) for s in ([0] if t == 0 else SEEDS)]
                rs = [r for r in rs if "gk" in r]
                if not rs or (kind in ("lc", "sv") and len(rs) < (1 if t == 0 else len(SEEDS))):
                    continue
                k, n = sum(r["gk"] for r in rs), sum(r["gn"] for r in rs)
                lo, hi = wilson(k, n)
                pts.append((t, 100 * k / n, lo, hi, t == 0 or len(rs) == len(SEEDS)))
            curves[(d, arm)] = np.array(pts, dtype=float).reshape(-1, 5)
    ymax = 5 * np.ceil(max(p[:, 1 if lines and not bars else 3].max() for p in curves.values() if len(p)) / 5 + 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    partial = False
    for ax, d in zip(axes, DS):
        for j, (arm, lab, col) in enumerate(SERIES):
            p = curves[(d, arm)].copy()
            if not len(p):
                continue
            if lines and bars:
                x = p[:, 0] + (j - 0.5) * 0.008 * K["end"][d]
                ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=col, ecolor=col,
                            lw=1.8, elinewidth=1.1, capsize=2.5, capthick=1.1, markersize=4.5, markeredgecolor=SURFACE,
                            markeredgewidth=0.8, label=lab)
                continue
            if lines:
                ax.plot(p[:, 0], p[:, 1], "-o", color=col, lw=1.8, markersize=4.5, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, label=lab)
                continue
            x = p[:, 0] + (j - 0.5) * 0.012 * K["end"][d]
            for full in (True, False):
                m = (p[:, 4] == 1) == full
                if m.any():
                    ax.errorbar(x[m], p[m, 1], yerr=[p[m, 1] - p[m, 2], p[m, 3] - p[m, 1]], fmt="o", color=col, ecolor=col,
                                elinewidth=1.5, capsize=3, capthick=1.5, markersize=6,
                                markeredgecolor=SURFACE if full else col, markerfacecolor=col if full else SURFACE,
                                markeredgewidth=1.2, label=lab if full else None)
                    partial = partial or not full
        ax.set_xlim(-0.04 * K["end"][d], K["end"][d] * 1.04)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(f"{DS[d]} fine-tune step")
        ax.set_title(DS[d], color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("general misalignment rate (%)")
    axes[0].legend(frameon=False, labelcolor=INK2, loc="upper left", fontsize=10)
    fig.suptitle(f"{TITLE[kind]}: EM-like vs random CPT documents (first-plot questions)"
                 + ("; open markers: fewer than 4 seeds" if partial else ""), color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out = ROOT / "results" / "attribution" / f"fig_cpt_fp3_{kind}{'_lines' if lines else ''}{'_bars' if lines and bars else ''}.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
