"""Run 61 heatmaps (em/mitigation/PREREG.md), from susceptibility_r61.py's numbers (same bootstrap draws):
results/mitigation/fig_r61_heatmaps.png. Rank-k LoRA on mlp.down_proj, ranks 1-256 (columns) x layer sets 1 / 5 / 12 /
48 (rows). Top row: susceptibility S (the sum of the headline over the 8 checkpoints, seed mean), no SDF and v5 side by
side on one colour scale; bottom row: the final-checkpoint headline (step 1125), the same way. Each cell shows the seed
mean and its 95% CI. No difference maps. SUSC_SEEDS=0 draws the seed-0 first look (fig_r61_heatmaps_seed0.png).
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r61.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r61 as sr  # noqa: E402

INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"   # as fig_r59.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.1})
CMAP = LinearSegmentedColormap.from_list("seq", ["#fbf3ef", "#f2b9a6", "#d9634f", "#a3243b", "#5c0f24"])  # one hue, light to dark
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
SUF = "" if sr.SEEDS == [0, 1, 2] else "_seed" + "".join(map(str, sr.SEEDS))
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
LAST = sr.LORA[-1]
YT = {1: "1 (24)", 5: "5 (16-32,\nevery 4th)", 12: "12 (0-44,\nevery 4th)", 48: "48 (all)"}


def grids(res, cv):
    """{(measure, arm): (values, lo, hi)} as layer-set x rank arrays (NaN where a cell is not judged)."""
    used, out = res["used"], res["out"]
    g = {}
    for m in sr.ARMS:
        for meas in ("S", "final"):
            v, lo, hi = (np.full((len(sr.LSETS), len(sr.RANKS)), np.nan) for _ in range(3))
            for i, (nl, _) in enumerate(sr.LSETS):
                for j, k in enumerate(sr.RANKS):
                    d = sr.GRID[nl, k]
                    if not used[d]:
                        continue
                    pt, bt = (out[m, d]["headline"][:2] if meas == "S" else (cv[m, d, LAST][0][0], cv[m, d, LAST][1][0]))
                    v[i, j], (lo[i, j], hi[i, j]) = pt, ci(bt)
            g[meas, m] = (v, lo, hi)
    return g


def panel(ax, v, lo, hi, vmax, title, fmt):
    ax.imshow(np.ma.masked_invalid(v), cmap=CMAP, vmin=0, vmax=vmax, origin="lower", aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0ec", edgecolor="white",
                                           hatch="///", linewidth=0))
                ax.text(j, i, "not\njudged", ha="center", va="center", fontsize=9, color=MUTED)
                continue
            dark = v[i, j] / vmax > 0.55
            ax.text(j, i + 0.1, fmt.format(v[i, j]), ha="center", va="center", fontsize=12.5, fontweight="bold",
                    color="white" if dark else INK)
            ax.text(j, i - 0.22, f"[{lo[i, j]:.0f}, {hi[i, j]:.0f}]", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK2)
    ax.set_xticks(range(len(sr.RANKS)))
    ax.set_xticklabels([str(k) for k in sr.RANKS])
    ax.set_yticks(range(len(sr.LSETS)))
    ax.set_yticklabels([YT[nl] for nl, _ in sr.LSETS])
    ax.set_xticks(np.arange(-0.5, len(sr.RANKS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sr.LSETS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("LoRA rank (down_proj)")
    ax.set_title(title, color=INK, loc="left")


def main():
    sp, res, cv = sr.run()
    used = res["used"]
    g = grids(res, cv)
    seeds = sorted(set(s for d in sr.CFG for s in used[d]))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), gridspec_kw=dict(hspace=0.42, wspace=0.12))
    for r, (meas, lab, fmt, step) in enumerate((("S", "susceptibility, the sum over 8 checkpoints",
                                                 "{:.0f}", 50),
                                                ("final", f"general misalignment at step {LAST} (%)",
                                                 "{:.1f}", 10))):
        vals = [g[meas, m][2] for m in sr.ARMS]
        vmax = np.ceil(np.nanmax(vals) / step) * step if np.isfinite(np.nanmax(vals)) else step
        for c, m in enumerate(sr.ARMS):
            v, lo, hi = g[meas, m]
            panel(axes[r, c], v, lo, hi, vmax, f"{sr.NAMES[m]}: {lab}", fmt)
            if c:
                axes[r, c].set_yticklabels([])
            else:
                axes[r, c].set_ylabel("layers with the LoRA")
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, ax=axes[r, :], fraction=0.025, pad=0.02)
        cb.set_label("susceptibility" if meas == "S" else "general misalignment (%)", color=INK2)
        cb.outline.set_visible(False)
    s0 = ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sr.ARMS)
    pre = "PRELIMINARY, seed 0 only (CI over questions only).  " if SUF else ""
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct, risky financial advice ({sd}): rank x layers of a down_proj LoRA, no SDF and v5 "
                 "(Run 61 with Runs 58 / 59)", x=0.06, y=1.0, ha="left", color=INK, fontsize=14)
    fig.text(0.06, 0.965, pre + "train_fast.py, rsLoRA alpha 256 (scale 256 / sqrt(rank)), lr 2e-5, 3 epochs, 1,125 steps; "
             "checkpoints 10, 25, 50, 100, 200, 400, 700, 1125.  v5 = full-parameter SDF on Run 54's demonstration "
             "corpus, grafted at 1.\nCells: " + ("seed 0" if SUF else "seed mean") + ", 95% CI below it; one colour scale per row.  Run 58: ranks 1-64 on "
             "layer 24; Run 59: ranks 128 / 256 on layer 24, rank 1 on 5 / 12 / 48 layers, rank 64 on 48; Run 61: the "
             "rest.  Every file on em/generate_vllm.py.\nSusceptibility: general misalignment (the em_eval_v2 chat headline, %) "
             "summed over the 8 checkpoints.  Step 0 (not in the sum): " + s0 + ".",
             ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / f"fig_r61_heatmaps{SUF}.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
