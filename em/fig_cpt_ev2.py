"""Figures: em_eval_v2 (76 chat questions x 10, report_cpt_ev2.py) on TOP (1B-CPT on EM-like documents) vs RANDOM
(1B-CPT on random documents) vs no CPT (stage 1), one figure per fine-tune dataset: before fine-tuning (one model per
arm), step 25 and the final model. Bars are the headline (the mean of the 6 category rates) averaged over seeds 0-3,
dots single seeds, error bars the sample SD over seeds. Colors as fig_cpt_fp3.py, with no CPT in light gray.

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
ARMS = [("TOP", "EM-like documents (top 1B)", "#2a78d6"), ("RANDOM", "random documents (1B)", "#7d7c76"),
        ("BASE", "no CPT", "#cfcdc4")]
DS = [("bad medical", "med", "Bad medical fine-tune", 441), ("risky finance", "fin", "Risky finance fine-tune", 375),
      ("extreme sports", "spo", "Extreme sports fine-tune", 375)]


def main():
    t = pd.read_csv(OUT / "cpt_ev2_per_model.csv")
    pre = t[t.step == 0].set_index("arm")["head"]
    ymax = 5 * (int(t["head"].max() // 5) + 2)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    for dname, tag, title, end in DS:
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        groups = [("before fine-tune", None), ("step 25", 25), (f"final (step {end})", end)]
        for i, (_, step) in enumerate(groups):
            for j, (arm, lab, col) in enumerate(ARMS):
                pos = i + (j - 1) * 0.27
                if step is None:
                    ax.bar(pos, pre[arm], 0.25, color=col, label=lab, zorder=2)
                    continue
                x = t[(t.dataset == dname) & (t.step == step) & (t.arm == arm)]["head"].to_numpy()
                ax.bar(pos, x.mean(), 0.25, color=col, zorder=2)
                ax.errorbar(pos, x.mean(), yerr=x.std(ddof=1), color=INK, lw=1, capsize=2.5, zorder=3)
                ax.scatter([pos] * len(x), x, s=10, color="white", edgecolor=INK, lw=0.6, zorder=4)
        ax.set_xticks(range(len(groups)), [g for g, _ in groups])
        ax.set_ylim(0, ymax)
        ax.set_ylabel("general misalignment (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
        ax.text(1, 1.01, "76 questions × 10 · 4 seeds · mean ± SD", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color=MUTED)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_cpt_ev2_{tag}.png", dpi=200)
        plt.close(fig)
        print(OUT / f"fig_cpt_ev2_{tag}.png")


if __name__ == "__main__":
    main()
