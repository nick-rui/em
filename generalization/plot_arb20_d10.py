"""Bar chart for the sp_arb20 10 x 10 design: P(held-out X chosen over held-out Y) after the outcome run (steering vector
at layer 12 trained on {x1..x10} over {y1..y10}), without and with SDF. Bar: mean over the 4 instances of each
instance's mean over its 100 held-out pairs (per-pair means as in report_arb.py); error bar: standard error across the 4
instances. Third bar, the baseline where X and Y are already associated: the four natural-category settings
(sp_colors, sp_fruits, sp_animals, sp_vehicles; 5 X and 5 Y items, untouched model, outcome run trained on only 2 pairs,
3 held-out pairs; results/generalization/<setting>/score_<setting>_outcome_L12.json).

  uv run --no-project --with matplotlib --with numpy python generalization/plot_arb20_d10.py
Writes results/generalization/fig_arb20_d10.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "generalization"
SLUGS = [f"sp_arb20_{i}" for i in range(1, 5)]
ARMS = {"Arbitrary sets,\nno SDF": "{s}_d10_outcome", "Arbitrary sets,\nSDF": "{s}_sdf_d10_outcome"}
NATURAL = ["sp_colors", "sp_fruits", "sp_animals", "sp_vehicles"]
COLORS = ["#8a8984", "#2a78d6", "#eb6834"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def heldout(name, s):
    acc = {}
    for r in json.load(open(R / s / f"score_{name}.json"))["items"]:
        if r["kind"] == "heldout":
            key = (r["target"], r["option_b"] if r["option_a"] == r["target"] else r["option_a"])
            acc.setdefault(key, []).append(r["p_target"])
    return [float(np.mean(v)) for v in acc.values()]


def main():
    inst = {a: [float(np.mean(heldout(n.format(s=s), s))) for s in SLUGS] for a, n in ARMS.items()}
    inst["Natural\ncategories"] = [float(np.mean(heldout(f"{s}_outcome_L12", s))) for s in NATURAL]
    vals = {a: float(np.mean(v)) for a, v in inst.items()}
    se = [float(np.std(v, ddof=1) / np.sqrt(len(v))) for v in inst.values()]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=200)
    ax.bar(range(len(vals)), vals.values(), width=0.5, color=COLORS, zorder=2, yerr=se,
           error_kw=dict(ecolor=INK, elinewidth=1.2, capsize=5, capthick=1.2, zorder=3))
    ax.set_xticks(range(len(vals)), vals.keys(), color=INK)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.set_title("Held-out X over held-out Y, after fine-tuning", fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout()
    out = R / "fig_arb20_d10.png"
    fig.savefig(out, facecolor="white")
    print("saved", out, {a.replace(chr(10), " "): (round(v, 3), round(e, 3)) for (a, v), e in zip(vals.items(), se)},
          {a.replace(chr(10), " "): [round(x, 3) for x in v] for a, v in inst.items()})


if __name__ == "__main__":
    main()
