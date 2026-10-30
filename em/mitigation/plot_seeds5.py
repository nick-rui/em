"""Figures for the graft-1 comparison (report_seed_stats.py's "graft1" set): one figure per fine-tune method, two panels
(general misalignment without quick_buck; domain misalignment), domains on the x axis. Bars are the mean over the seeds
judged in every arm, dots are single seeds, error bars the sample SD. No SDF gray, Dolma graft 1 orange, the
narrow-character prior blue.

  uv run --no-project --with pandas --with scipy python em/mitigation/report_seed_stats.py \\
      > results/mitigation/report_seed_stats.txt
  uv run --no-project --with pandas --with matplotlib python em/mitigation/plot_seeds5.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ARMS = [("no SDF", "no SDF", "#8a8a8a"), ("Dolma graft 1", "Dolma SDF", "#e8890c"),
        ("prior graft 1", "narrow-character SDF", "#2f6db5")]
DOMS = ["medical", "finance", "sports"]
PANELS = [("general", "general misalignment (%)"), ("domain", "in-domain misalignment (%)")]
METHODS = {"full": ("full", "Full fine-tune"), "rank 32": ("r32", "Rank-32 LoRA"), "rank 1": ("r1", "Rank-1 LoRA")}


def main():
    t = pd.read_csv(OUT / "seed_stats_per_question.csv")
    t = t[t.set == "graft1"]
    seed = t.groupby(["domain", "method", "arm", "measure", "seed"])[["k", "n"]].sum().reset_index()
    seed["rate"] = 100 * seed.k / seed.n
    for meth, (tag, title) in METHODS.items():
        s = seed[seed.method == meth]
        if s.empty:
            continue
        seeds = sorted(s.seed.unique())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, (measure, ylab) in zip(axes, PANELS):
            for j, (arm, lab, col) in enumerate(ARMS):
                for i, d in enumerate(DOMS):
                    x = s[(s.domain == d) & (s.arm == arm) & (s.measure == measure)].rate.to_numpy()
                    if not len(x):
                        continue
                    pos = i + (j - 1) * 0.26
                    ax.bar(pos, x.mean(), 0.24, color=col, label=lab if i == 0 else None, zorder=2)
                    ax.errorbar(pos, x.mean(), yerr=x.std(ddof=1) if len(x) > 1 else 0, color="black", lw=1,
                                capsize=2.5, zorder=3)
                    ax.scatter([pos] * len(x), x, s=9, color="white", edgecolor="black", lw=0.6, zorder=4)
            ax.set_xticks(range(len(DOMS)), DOMS)
            ax.set_ylabel(ylab)
            ax.set_ylim(0, 100 if measure == "domain" else None)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=0)
        axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2,
                       columnspacing=1.0, borderaxespad=0.2)
        axes[1].text(1, 1.02, f"graft 1 · {len(seeds)} seeds · mean ± SD", transform=axes[1].transAxes,
                     ha="right", va="bottom", fontsize=8, color="#555555")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_seeds5_{tag}.png", dpi=200)
        plt.close(fig)
        print(f"{title}: seeds {seeds} -> {OUT / f'fig_seeds5_{tag}.png'}")


if __name__ == "__main__":
    main()
