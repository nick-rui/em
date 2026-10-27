"""Bar chart for run 37 (em/mitigation/PREREG.md): general and sports misalignment after the extreme-sports rank-1
fine-tune (layer 24, seed 0), for no SDF, the v1 sports prior at graft 1 (run 13) and the rebuilt 50M-token prior at
grafts 0.5 and 1. Error bars are the binomial standard error over responses (one seed). Same definitions as
report_sports_v2.py.

  uv run --no-project --with matplotlib --with pandas --with scipy python em/mitigation/plot_sports_v2.py
Writes results/mitigation/fig_sports_v2.png.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_sports_v2 as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
MODELS = [("No SDF", "mit_q14i_spo2_s0"), ("Old sports SDF\n(20M, graft 1)", "mit_q14i_graftspo1_spo"),
          ("New sports SDF\n(50M, graft 0.5)", "mit_q14i_graftspo2_0p5_spo_s0"),
          ("New sports SDF\n(50M, graft 1)", "mit_q14i_graftspo2_1_spo_s0")]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    w, gap = 0.36, 0.02
    rs = [Q.rates(n) for _, n in MODELS]
    for k, (kk, nn) in enumerate((("k", "n"), ("km", "nm"))):
        xs = [i + (k - 0.5) * (w + gap) for i in range(len(MODELS))]
        ps = [r[kk] / r[nn] if r else 0 for r in rs]
        es = [100 * math.sqrt(p * (1 - p) / r[nn]) if r else 0 for p, r in zip(ps, rs)]
        ax.bar(xs, [100 * p for p in ps], width=w, color=COLORS[k], zorder=2)
        ax.errorbar(xs, [100 * p for p in ps], yerr=es, fmt="none", ecolor=INK, elinewidth=1.0, capsize=3,
                    capthick=1.0, zorder=3)
    ax.set_xticks(range(len(MODELS)), [lab for lab, _ in MODELS], color=INK, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{y}%" for y in range(0, 101, 25)], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    fig.legend(handles=[Patch(color=COLORS[0], label="General misalignment"),
                        Patch(color=COLORS[1], label="Sports misalignment")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.suptitle("Narrow-assistant SDF + Rank-1 Fine-Tuned on Extreme Sports (Layer 24)", fontsize=12, color=INK,
                 y=0.99)
    out = ROOT / "results" / "mitigation" / "fig_sports_v2.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
