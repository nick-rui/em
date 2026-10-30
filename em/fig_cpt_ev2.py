"""Figure: em_eval_v2 (76 chat questions x 10, report_cpt_ev2.py) on TOP (1B-CPT on EM-like documents) vs RANDOM
(1B-CPT on random documents) vs no CPT (stage 1), in fig_cpt_fp3.py's style, one panel per fine-tune dataset: general
misalignment (the headline, the mean of the 6 category rates) against the fine-tune step, on a symmetric-log step axis
(linear below 10, so step 0 fits). Step 0 is the model before fine-tuning (one per arm, no error bar); step 25 and the
end are the mean over seeds 0-3 with the sample SD over seeds. No CPT, the diagnostic arm, is dashed with open markers.

  uv run --no-project --with pandas --with matplotlib python em/fig_cpt_ev2.py   # -> results/attribution/fig_cpt_ev2.png
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "attribution"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARMS = [("TOP", "EM-like documents (top 1B)", "#2a78d6", "-"), ("RANDOM", "random documents (1B)", "#7d7c76", "-"),
        ("BASE", "no CPT", "#a9a79e", "--")]
DS = [("bad medical", 441), ("risky finance", 375), ("extreme sports", 375)]


def main():
    t = pd.read_csv(OUT / "cpt_ev2_per_model.csv")
    pre = t[t.step == 0].set_index("arm")["head"]
    ft = t[t.step > 0].groupby(["dataset", "step", "arm"])["head"].agg(["mean", "std"])
    ymax = 5 * (int((ft["mean"] + ft["std"]).max() // 5) + 2)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (dname, end) in zip(axes, DS):
        for j, (arm, lab, col, ls) in enumerate(ARMS):
            steps = [0, 25, end]
            y = [pre[arm]] + [ft.loc[(dname, s, arm), "mean"] for s in steps[1:]]
            e = [0] + [ft.loc[(dname, s, arm), "std"] for s in steps[1:]]
            x = [(j - 1) * 0.6] + [s * (1 + (j - 1) * 0.07) for s in steps[1:]]   # small offsets so bars don't overlap
            ax.errorbar(x, y, yerr=e, fmt="o", ls=ls, color=col, ecolor=col, lw=1.8, elinewidth=1.1, capsize=2.5,
                        capthick=1.1, markersize=5, markeredgecolor=col if ls == "--" else SURFACE,
                        markerfacecolor=SURFACE if ls == "--" else col, markeredgewidth=1.2 if ls == "--" else 0.8,
                        label=lab)
        ax.set_xscale("symlog", linthresh=10, linscale=0.4)
        ax.set_xlim(-2, end * 1.25)
        ax.set_xticks([0, 10, 25, 100, end], ["0", "10", "25", "100", str(end)])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, ymax)
        ax.set_xlabel(f"{dname} fine-tune step")
        ax.set_title(dname, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("general misalignment rate (%)")
    axes[0].legend(frameon=False, labelcolor=INK2, loc="lower right", fontsize=9)
    axes[-1].text(1, 1.01, "76 questions × 10 · 4 seeds · mean ± SD", transform=axes[-1].transAxes, ha="right",
                  va="bottom", fontsize=8, color=MUTED)
    fig.suptitle("Full fine-tune: EM-like vs random CPT documents (em_eval_v2)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cpt_ev2.png", dpi=200)
    print(OUT / "fig_cpt_ev2.png")


if __name__ == "__main__":
    main()
