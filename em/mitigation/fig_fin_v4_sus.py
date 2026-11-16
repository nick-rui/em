"""Run 52 figure (em/mitigation/PREREG.md): em_eval_v2 chat headline (% misaligned, gpt-6-luna) against EM fine-tuning
step on risky financial advice, one panel per recipe (rank 1 layer 24 on train_fast.py; full fine-tune on train.py), the
three arms: no SDF, S full-parameter SDF (statement docs), D full-parameter SDF (demonstration docs). Lines are seed
means (seeds 0-2); bands are 95% bootstrap intervals over questions within category and seed IDs (the draws of
susceptibility_fin_v4.py). The legend gives each arm's susceptibility to EM, the sum of its points in that panel
(PREREG 5.1, nothing subtracted, step 0 not in the sum); sums compare only within a panel. Step 0 (before fine-tuning)
is in the panel subtitle for reference.
  ~/.venvs/em-judge/bin/python em/mitigation/fig_fin_v4_sus.py   (with matplotlib)
  or: uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_fin_v4_sus.py
  -> results/mitigation/fig_fin_v4_sus.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_fin_v4 as sf  # noqa: E402

COL = {"nosdf": "#8a8984", "S": "#2a78d6", "D": "#eb6834"}
SHORT = {"nosdf": "no SDF", "S": "S", "D": "D"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_ev2_mit_full.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation" / "fig_fin_v4_sus.png"


def main():
    sp = sf.spec()
    res = sf.sc.compute(sp)
    used = res["used"]
    cv = sf.curves(sp, used)
    ds = [d for d in sf.RECIPES if used[d]]
    fig, axes = plt.subplots(1, len(ds), figsize=(6.2 * len(ds), 4.8), sharey=True, squeeze=False)
    ymax = max(np.percentile(cv[m, d, t][1][0], 97.5) for d in ds for m in sf.ARMS for t in sf.STEPS[d])
    for ax, d in zip(axes[0], ds):
        x = np.array(sf.STEPS[d])
        for m in sf.ARMS:
            pt = np.array([cv[m, d, t][0][0] for t in x])
            lo, hi = np.array([np.percentile(cv[m, d, t][1][0], [2.5, 97.5]) for t in x]).T
            S = res["out"][m, d]["headline"]
            Slo, Shi = np.percentile(S[1], [2.5, 97.5])
            ax.fill_between(x, lo, hi, color=COL[m], alpha=0.14, linewidth=0)
            ax.plot(x, pt, color=COL[m], linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                    markeredgewidth=1.2, label=f"{SHORT[m]}: {S[0]:.0f} [{Slo:.0f}, {Shi:.0f}]")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in x])
        ax.minorticks_off()
        ax.set_xlim(x[0] / 1.35, x[-1] * 1.35)
        ax.set_ylim(0, max(70, np.ceil(ymax / 10) * 10))
        ax.set_xlabel("EM fine-tuning step (log scale)")
        k = len(x)
        ax.set_title(f"{sf.RECIPES[d]}, {k} checkpoints", color=INK, loc="left", pad=20)
        step0 = ", ".join(f"{SHORT[m]} {cv[m, None, 0][0][0]:.1f}" for m in sf.ARMS)
        ax.text(0, 1.015, f"step 0 (before fine-tuning, not in the sum): {step0}", transform=ax.transAxes,
                fontsize=9.5, color=MUTED, va="bottom")
        ax.legend(loc="lower right", title=f"susceptibility to EM\n(sum of the {k} points) [95% CI]",
                  title_fontsize=9.5, handlelength=1.6, alignment="left")
        ax.get_legend().get_title().set_color(INK2)
    axes[0][0].set_ylabel("em_eval_v2 headline (% misaligned)")
    seeds = sorted(set(s for d in ds for s in used[d]))
    fig.suptitle("Qwen2.5-14B-Instruct, risky financial advice: misalignment during EM fine-tuning "
                 f"(seed mean, seeds {', '.join(map(str, seeds))})", x=0.01, ha="left", color=INK, fontsize=13.5, y=1.07)
    fig.text(0.01, 1.01, "S = full-parameter SDF on statement docs;  D = full-parameter SDF on demonstration "
             "docs (both grafted at 1).  Bands: 95% bootstrap over questions within category and seeds.",
             ha="left", color=INK2, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()
