"""Figure: general misalignment before any fine-tune, TOP (1B-CPT on EM-like documents) vs RANDOM (1B-CPT on random
documents), on Betley et al.'s 48 questions x 100 (generation seeds 0 and 1), Wilson 95% CIs (em/TODO.md).
  python em/fig_cpt_bar_preft.py   # -> results/attribution/fig_cpt_bar_preft.png
  python em/fig_cpt_bar_preft.py --fp   # first-plot's 8 questions x 50 (generation seed 2; the learning curves' step 0,
                                        # cptlc_<arm>_t000) -> results/attribution/fig_cpt_bar_preft_fp.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "scripts"))
import report_cpt_prereg48 as rep48  # noqa: E402
from report_cpt_lc import counts  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARMS = [("EM-like documents\n(top 1B)", ["cpt48", "cpt48b"], "#2a78d6"),
        ("random documents\n(1B)", ["cptr48", "cptr48b"], "#7d7c76")]


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    fp = "--fp" in sys.argv
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, (lab, prefixes, col) in enumerate(ARMS):
        if fp:
            r = counts(f"cptlc_{'top1b' if i == 0 else 'rand1b'}_t000")
            k, n = r["gk"], r["gn"]
        else:
            rep48.PREFIXES = prefixes
            d = rep48.load()
            g = d[d.dose == 10]        # the 1B-token checkpoint
            k, n = g.mis.sum(), len(g)
        lo, hi = rep48.wilson(k, n)
        p = 100 * k / n
        ax.bar(i, p, width=0.6, color=col)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK2, elinewidth=1.5, capsize=5, capthick=1.5)
        print(f"{lab.replace(chr(10), ' ')}: {p:.2f}% [{lo:.1f}-{hi:.1f}], n {n}")
    ax.set_xticks([0, 1], [a[0] for a in ARMS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 12)
    ax.set_ylabel("general misalignment rate (%)")
    ax.set_title(f"Before fine-tuning ({'first-plot, 8 questions' if fp else 'Betley' + chr(39) + 's 48 questions'})",
                 color=INK, fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = ROOT / "results" / "attribution" / f"fig_cpt_bar_preft{'_fp' if fp else ''}.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
