"""Bar chart for the swap test in the mechanistic analysis (generalization/mech_arb20.py, check (a)): P(held-out X chosen
over held-out Y) on eval10 for the 10 x 10 fine-tuned vector trained without SDF (grey) and with SDF (blue), each
run in the no-SDF model and in the SDF model. Per instance the two seeds (0, 1) are averaged; bar: mean over the 4
arbitrary instances; error bar: standard error across them.

  uv run --no-project --with matplotlib --with numpy python generalization/plot_mech_swap.py
Writes results/generalization/fig_mech_swap.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "generalization"
SLUGS = [f"sp_arb20_{i}" for i in range(1, 5)]
# (model the vector runs in) -> {vector: score name}
CELLS = {"Run in the no-SDF model": {"Trained without SDF": ["{s}_d10_outcome", "{s}_d10_outcome_s1"],
                                    "Trained with SDF": ["mech_{s}_swap_sdf_on_nosdf", "mech_{s}_s1_swap_sdf_on_nosdf"]},
         "Run in the SDF model": {"Trained without SDF": ["mech_{s}_swap_nosdf_on_sdf", "mech_{s}_s1_swap_nosdf_on_sdf"],
                                 "Trained with SDF": ["{s}_sdf_d10_outcome", "{s}_sdf_d10_outcome_s1"]}}
COLORS = {"Trained without SDF": "#8a8984", "Trained with SDF": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def heldout(names, s):
    """mean over seeds of P(held-out X over held-out Y)"""
    return float(np.mean([json.load(open(R / s / f"score_{n.format(s=s)}.json"))["summary"]["heldout"] for n in names]))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
    w, gap = 0.34, 0.02
    for k, vname in enumerate(COLORS):
        xs, ms, es = [], [], []
        for i, (model, cell) in enumerate(CELLS.items()):
            v = np.array([heldout(cell[vname], s) for s in SLUGS])
            xs.append(i + (k - 0.5) * (w + gap))
            ms.append(v.mean())
            es.append(v.std(ddof=1) / np.sqrt(len(v)))
            print(f"{model:24s} {vname:20s} {v.mean():.3f} +- {es[-1]:.3f}  {np.round(v, 3).tolist()}")
        ax.bar(xs, ms, width=w, color=COLORS[vname], label=vname, zorder=2, yerr=es,
               error_kw=dict(ecolor=INK, elinewidth=1.2, capsize=5, capthick=1.2, zorder=3))
    ax.set_xticks(range(len(CELLS)), CELLS.keys(), color=INK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK, ncol=2)
    ax.set_title("Held-out X over held-out Y, vectors swapped", fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout()
    out = R / "fig_mech_swap.png"
    fig.savefig(out, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
