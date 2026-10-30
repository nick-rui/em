"""Figures: em_eval_v2 (76 chat questions x 10, report_cpt_ev2.py) on TOP (1B-CPT on EM-like documents) vs RANDOM
(1B-CPT on random documents) vs no CPT (stage 1), one figure per fine-tune dataset, in fig_cpt_fp3.py's style: general
misalignment (the headline, the mean of the 6 category rates) against the fine-tune step. Step 0 is the model before
fine-tuning (one per arm, no error bar); steps 25 and the end are the mean over seeds 0-3 with the sample SD over seeds.
No CPT, the diagnostic arm, is dashed with open markers.

  uv run --no-project --with pandas --with matplotlib python em/fig_cpt_ev2.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "attribution"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARMS = [("TOP", "EM-like documents (top 1B)", "#2a78d6", "-"), ("RANDOM", "random documents (1B)", "#7d7c76", "-"),
        ("BASE", "no CPT", "#a9a79e", "--")]
DS = [("bad medical", "med", "Bad medical fine-tune", 441), ("risky finance", "fin", "Risky finance fine-tune", 375),
      ("extreme sports", "spo", "Extreme sports fine-tune", 375)]


def main():
    t = pd.read_csv(OUT / "cpt_ev2_per_model.csv")
    pre = t[t.step == 0].set_index("arm")["head"]
    ft = t[t.step > 0].groupby(["dataset", "step", "arm"])["head"].agg(["mean", "std"])
    ymax = 5 * (int((ft["mean"] + ft["std"]).max() // 5) + 2)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    for dname, tag, title, end in DS:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        for j, (arm, lab, col, ls) in enumerate(ARMS):
            steps = [0, 25, end]
            y = [pre[arm]] + [ft.loc[(dname, s, arm), "mean"] for s in steps[1:]]
            e = [0] + [ft.loc[(dname, s, arm), "std"] for s in steps[1:]]
            x = [s + (j - 1) * 0.012 * end for s in steps]
            ax.errorbar(x, y, yerr=e, fmt="o", ls=ls, color=col, ecolor=col, lw=1.8, elinewidth=1.1, capsize=2.5,
                        capthick=1.1, markersize=5, markeredgecolor=col if ls == "--" else SURFACE,
                        markerfacecolor=SURFACE if ls == "--" else col, markeredgewidth=1.2 if ls == "--" else 0.8,
                        label=lab)
        ax.set_xlim(-0.04 * end, end * 1.04)
        ax.set_xticks([0, 25, 100, 200, 300, end])
        ax.set_ylim(0, ymax)
        ax.set_xlabel(f"{dname} fine-tune step")
        ax.set_ylabel("general misalignment rate (%)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, labelcolor=INK2, loc="lower right", fontsize=9)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.text(1, 1.01, "76 questions × 10 · 4 seeds · mean ± SD", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color=MUTED)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_cpt_ev2_{tag}.png", dpi=200)
        plt.close(fig)
        print(OUT / f"fig_cpt_ev2_{tag}.png")


if __name__ == "__main__":
    main()
