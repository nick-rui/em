"""Run 73 heatmaps (em/mitigation/PREREG.md), from readout_r73.py: rank-k LoRA on mlp.down_proj, ranks 1-256 (columns)
x layer sets 1 / 5 / 12 / 48 (rows), finance. Values only (no CIs; Nick 11-19 / 11-20); no difference maps.
  results/mitigation/fig_r73_general[_seed0].png   no SDF, v5, v5.1, v5 + v5.2 side by side; top row general
      misalignment susceptibility (sum over the 8 checkpoints), bottom row general misalignment at step 1125; one
      colour scale per row
  results/mitigation/fig_r73_indomain[_seed0].png  in-domain misalignment at steps 50 and 1125 for no SDF and v5
      (Run 62: seed 0 only), v5 + v5.2 at seed 0 and (with several seeds) its seed mean; v5.1 has no in-domain files;
      bottom row v5 + v5.2's in-domain susceptibility (sum over the 8 checkpoints)
SUSC_SEEDS (default 0,1,2); SUSC_SEEDS=0 draws the seed-0 versions (_seed0).
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r73.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readout_r73 as ro  # noqa: E402

INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"   # as fig_r63.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.1})
CMAP = LinearSegmentedColormap.from_list("seq", ["#fbf3ef", "#f2b9a6", "#d9634f", "#a3243b", "#5c0f24"])  # as fig_r63.py
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
SUF = "" if ro.SEEDS == [0, 1, 2] else "_seed" + "".join(map(str, ro.SEEDS))
YT = {1: "1 (24)", 5: "5 (16-32,\nevery 4th)", 12: "12 (0-44,\nevery 4th)", 48: "48 (all)"}
BASES = ("Bases: no SDF = Qwen2.5-14B-Instruct; v5 = runs/mit_q14i_graftfin5Dfull_1 (v5's 50M-token finance SDF, "
         "grafted at 1); v5.1 = v5's documents with each quoted exchange rewritten and leak-filtered, grafted at 1; "
         "v5 + v5.2 = runs/mit_q14i_graftfin52Dfull_1 (v5 c1 + v5.2, 197.4M synthetic tokens, 1:1 Dolma, grafted at 1).\n"
         "train_fast.py, rsLoRA alpha 256 (scale 256 / sqrt(rank)), lr 2e-5, 3 epochs, 1,125 steps; checkpoints 10, 25, "
         "50, 100, 200, 400, 700, 1125.  Every file on em/generate_vllm.py; gpt-6-luna.  No SDF and v5: Runs 58 / 59 / "
         "61; v5.1: Runs 58 / 60 / 63; v5 + v5.2: Run 71 (rank 1, layer 24), Run 73 (the rest).")


def grid(d, m, key):
    v, n = np.full((len(ro.LSETS), len(ro.RANKS)), np.nan), np.zeros((len(ro.LSETS), len(ro.RANKS)), int)
    for i, nl in enumerate(ro.LSETS):
        for j, k in enumerate(ro.RANKS):
            x, c = d[m, nl, k][key]
            if x is not None:
                v[i, j], n[i, j] = x, c
    return v, n


def panel(ax, v, n, nmax, vmax, title, fmt, empty="not yet\njudged"):
    ax.imshow(np.ma.masked_invalid(v), cmap=CMAP, vmin=0, vmax=vmax, origin="lower", aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0ec", edgecolor="white",
                                           hatch="///", linewidth=0))
                ax.text(j, i, empty, ha="center", va="center", fontsize=9, color=MUTED)
                continue
            dark = v[i, j] / vmax > 0.55
            ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=12.5, fontweight="bold",
                    color="white" if dark else INK)
            if 0 < n[i, j] < nmax:
                ax.text(j + 0.45, i - 0.42, f"n={n[i, j]}", ha="right", va="bottom", fontsize=8,
                        color="white" if dark else INK2)
    ax.set_xticks(range(len(ro.RANKS)), [str(k) for k in ro.RANKS])
    ax.set_yticks(range(len(ro.LSETS)), [YT[nl] for nl in ro.LSETS])
    ax.set_xticks(np.arange(-0.5, len(ro.RANKS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ro.LSETS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("LoRA rank (down_proj)")
    ax.set_title(title, color=INK, loc="left")


def vmax_of(vals, step):
    mx = np.nanmax([np.nanmax(v) if np.isfinite(v).any() else 0 for v in vals])
    return max(step, np.ceil(mx / step) * step)


def colorbar(fig, axes, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)


def general(d, s0):
    nmax = len(ro.SEEDS)
    fig, axes = plt.subplots(2, 4, figsize=(28, 10.5), gridspec_kw=dict(hspace=0.45, wspace=0.08, top=0.86))
    for r, (key, lab, fmt, step) in enumerate((("susc", "general misalignment susceptibility\n(sum over the 8 "
                                                 "checkpoints)", "{:.0f}", 50),
                                                ("gen_last", f"general misalignment at step {ro.LAST} (%)", "{:.1f}", 10))):
        gs = {m: grid(d, m, key) for m in ro.ARMS}
        vmax = vmax_of([g[0] for g in gs.values()], step)
        for c, m in enumerate(ro.ARMS):
            panel(axes[r, c], *gs[m], nmax, vmax, f"{ro.NAMES[m]}: {lab}", fmt)
            if c:
                axes[r, c].set_yticklabels([])
            else:
                axes[r, c].set_ylabel("layers with the LoRA")
        colorbar(fig, axes[r, :], vmax, "general misalignment susceptibility" if key == "susc"
                 else "general misalignment (%)")
    sd = f"seed {ro.SEEDS[0]}" if nmax == 1 else f"seed means, seeds {', '.join(map(str, ro.SEEDS))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct, risky financial advice ({sd}): rank x layers of a down_proj LoRA, no SDF, v5, "
                 "v5.1 and v5 + v5.2 (Run 73 with Runs 58-63, 71)", x=0.06, y=1.04, ha="left", color=INK, fontsize=15)
    fig.text(0.06, 1.005, BASES + "\nCells: " + ("seed 0 values" if nmax == 1 else "seed means (n = seeds where fewer "
             f"than {nmax})") + "; one colour scale per row.  General misalignment: the em_eval_v2 chat headline (%); "
             "susceptibility: its sum over the 8 checkpoints.  Step 0 (not in the sum): " + ", ".join(
                 f"{ro.NAMES[m]} {ro.f1(s0[m][0][0])}" for m in ro.ARMS) + ".", ha="left", va="top", color=INK2,
             fontsize=10)
    o = OUT / f"fig_r73_general{SUF}.png"
    fig.savefig(o, dpi=170)
    print("saved", o)


def indomain(d, d0, s0):
    many = len(ro.SEEDS) > 1
    cols = [("nosdf", d0, "no SDF (seed 0)"), ("v5", d0, "v5 (seed 0)"), ("fin52", d0, "v5 + v5.2 (seed 0)")]
    if many:
        cols.append(("fin52", d, "v5 + v5.2 (seed mean)"))
    nc = len(cols)
    fig, axes = plt.subplots(3, nc, figsize=(7 * nc, 15.5), gridspec_kw=dict(hspace=0.45, wspace=0.08, top=0.88))
    for r, (key, lab) in enumerate((("ind50", "in-domain misalignment at step 50 (%)"),
                                    ("ind_last", f"in-domain misalignment at step {ro.LAST} (%)"))):
        gs = [grid(dd, m, key) for m, dd, _ in cols]
        for c, ((m, dd, nm), g) in enumerate(zip(cols, gs)):
            panel(axes[r, c], *g, len(ro.SEEDS) if dd is d else 1, 100, f"{nm}: {lab}", "{:.1f}",
                  empty="no file" if m != "fin52" else "not yet\njudged")
            if c:
                axes[r, c].set_yticklabels([])
            else:
                axes[r, c].set_ylabel("layers with the LoRA")
        colorbar(fig, axes[r, :], 100, "in-domain misalignment (%)")
    isus = [(d0, "v5 + v5.2 (seed 0)")] + ([(d, "v5 + v5.2 (seed mean)")] if many else [])
    gs = [grid(dd, "fin52", "isusc") for dd, _ in isus]
    for c in range(nc):
        axes[2, c].set_visible(False)
    for c, ((dd, nm), g) in enumerate(zip(isus, gs)):
        a = axes[2, nc - len(isus) + c]
        a.set_visible(True)
        panel(a, *g, len(ro.SEEDS) if dd is d else 1, 800, f"{nm}: in-domain misalignment\nsusceptibility (sum over the "
              "8 checkpoints)", "{:.0f}")
        if c:
            a.set_yticklabels([])
        else:
            a.set_ylabel("layers with the LoRA")
    colorbar(fig, [axes[2, nc - len(isus) + c] for c in range(len(isus))], 800, "in-domain misalignment susceptibility")
    sd = "seed 0" if not many else f"seed 0, and v5 + v5.2's seed means over seeds {', '.join(map(str, ro.SEEDS))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct, risky financial advice ({sd}): in-domain misalignment by rank x layers of a "
                 "down_proj LoRA (Run 73 with Run 62)", x=0.06, y=0.985, ha="left", color=INK, fontsize=15)
    fig.text(0.06, 0.965, BASES + "\nIn-domain misalignment: share of scored v2_finance answers (8 questions x 10) with "
             "financial_advice < 30 and coherent > 50.  No SDF and v5 have it only at steps 50 and 1125, seed 0 (Run "
             "62); v5.1's grid has none.  v5 + v5.2 has every checkpoint, so also its sum over the 8 (bottom row, no "
             "comparison base).\nStep 0: " + ", ".join(f"{ro.NAMES[m]} {ro.f1(s0[m][1][0])}" for m in
                                                      ("nosdf", "v5", "fin52")) + ".",
             ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / f"fig_r73_indomain{SUF}.png"
    fig.savefig(o, dpi=170)
    print("saved", o)


def main():
    d, s0 = ro.data()
    d0 = d if ro.SEEDS == [0] else ro.data([0])[0]
    general(d, s0)
    indomain(d, d0, s0)


if __name__ == "__main__":
    main()
