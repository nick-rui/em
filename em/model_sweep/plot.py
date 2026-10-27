"""Bar chart for EM model sweep run 1 (em/model_sweep/PREREG.md): for each model, general misalignment (first-plot)
and medical misalignment (medical_questions, the fine-tune's own domain) after the bad-medical rank-32 fine-tune,
seed 0. Error bars are the binomial standard error over responses. Models are sorted by general misalignment. Same
definitions as report.py; models not judged yet are left out.

  uv run --no-project --with matplotlib --with pandas python em/model_sweep/plot.py
Writes results/model_sweep/fig_run1.png.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
MEASURES = [("gen", "General misalignment"), ("med", "Medical misalignment (in-domain)")]


def stats(kn):
    k, n = kn
    p = k / n
    return 100 * p, 100 * math.sqrt(p * (1 - p) / n)


def main():
    done, missing = Q.table()
    done = done[::-1]                         # barh draws bottom-up: highest general misalignment on top
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(done) + 1.6), dpi=200)
    h, gap = 0.36, 0.04
    for k, (key, _) in enumerate(MEASURES):
        ys = [i + (0.5 - k) * (h + gap) for i in range(len(done))]
        st = [stats(r["ft"][key]) for r in done]
        ax.barh(ys, [m for m, _ in st], height=h, color=COLORS[k], zorder=2)
        ax.errorbar([m for m, _ in st], ys, xerr=[e for _, e in st], fmt="none", ecolor=INK, elinewidth=1.0,
                    capsize=2.5, capthick=1.0, zorder=3)
    ax.set_yticks(range(len(done)), [r["name"] for r in done], color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25), [f"{x}%" for x in range(0, 101, 25)], color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.set_ylim(-0.6, len(done) - 0.4)
    ax.set_title("Emergent misalignment after the bad-medical fine-tune", fontsize=13, color=INK, pad=10, loc="left")
    ax.legend(handles=[Patch(color=c, label=lab) for c, (_, lab) in zip(COLORS, MEASURES)], loc="upper center",
              bbox_to_anchor=(0.4, -0.06 if len(done) > 6 else -0.15), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK)
    fig.tight_layout()
    out = ROOT / "results" / "model_sweep" / "fig_run1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out, f"({len(done)} models; not judged yet: {', '.join(r['name'] for r in missing) or 'none'})")


if __name__ == "__main__":
    main()
