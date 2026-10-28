"""Bar chart for run 40 (em/mitigation/PREREG.md): general and finance misalignment after the risky_financial rank-1
fine-tune (layer 24), mean over seeds 0-4, for no SDF and the v2 finance prior at grafts 0.5 and 1. General
misalignment leaves out quick_buck. Error bars are the standard error over the 5 seeds; dots are the seeds. Same
definitions as report_finance_v2_seeds.py.

  uv run --no-project --with matplotlib --with pandas --with scipy python em/mitigation/plot_finance_v2_seeds.py
Writes results/mitigation/fig_finance_v2_seeds.png.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_finance_v2 as Q  # noqa: E402
from report_finance_v2_seeds import SEEDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
MODELS = [("No SDF", "mit_q14i_fin2_s{}"), ("New finance SDF\n(50M, graft 0.5)", "mit_q14i_graftfin2_0p5_fin_s{}"),
          ("New finance SDF\n(50M, graft 1)", "mit_q14i_graftfin2_1_fin_s{}")]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    w, gap = 0.36, 0.02
    rs = [[Q.rates(f.format(s), drop=Q.NEAR) for s in SEEDS] for _, f in MODELS]
    for k, (kk, nn) in enumerate((("k", "n"), ("km", "nm"))):
        xs = [i + (k - 0.5) * (w + gap) for i in range(len(MODELS))]
        per = [[100 * r[kk] / r[nn] for r in m] for m in rs]
        mean = [sum(p) / len(p) for p in per]
        se = [math.sqrt(sum((x - mu) ** 2 for x in p) / (len(p) - 1) / len(p)) for p, mu in zip(per, mean)]
        print(kk, [f"{mu:.1f} ± {e:.1f}" for mu, e in zip(mean, se)])
        ax.bar(xs, mean, width=w, color=COLORS[k], zorder=2)
        ax.errorbar(xs, mean, yerr=se, fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
        for x, p in zip(xs, per):
            ax.scatter([x + d for d in (-0.08, -0.04, 0, 0.04, 0.08)], p, s=12, color=INK2, edgecolors=SURFACE,
                       linewidths=0.8, zorder=4)
    ax.set_xticks(range(len(MODELS)), [lab for lab, _ in MODELS], color=INK, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{y}%" for y in range(0, 101, 25)], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    fig.legend(handles=[Patch(color=COLORS[0], label="General misalignment (without quick_buck)"),
                        Patch(color=COLORS[1], label="Finance misalignment")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.suptitle("Narrow-assistant SDF + Rank-1 Fine-Tuned on Risky Finance (Layer 24, 5 Seeds)", fontsize=11.5,
                 color=INK, y=0.99)
    out = ROOT / "results" / "mitigation" / "fig_finance_v2_seeds.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
