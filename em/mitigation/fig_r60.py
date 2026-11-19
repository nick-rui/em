"""Run 60 figures (em/mitigation/PREREG.md), from susceptibility_r60.py's numbers (same bootstrap draws): Run 59's
figures (fig_r59.py) with v5.1 as a third arm.
  rank:    results/mitigation/fig_r60_rank.png     (a) S against LoRA rank 1, 16, 32, 64, 128, 256 on layer 24 down_proj
  layers:  results/mitigation/fig_r60_layers.png   (b) S against the number of layers (1, 5, 12, 48) at rank 1
  bridges: results/mitigation/fig_r60_bridges.png  (c) rank 1 on layer 24 / every layer, rank 64 on every layer (k = 8),
                                                       then all-module rank 32 and full (k = 6)
Each: top, S per config, seed mean with 95% CI, the three arms side by side; bottom, the headline against fine-tuning
step per config (seed means, 95% bands), the sum [95% CI] under each panel. Colours as fig_r58.py (no SDF grey dashed,
v5 magenta, v5.1 violet). SUSC_SEEDS=0 draws seed 0 only (files *_seed0.png).
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r60.py [rank|layers|bridges]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r60 as sr  # noqa: E402

COL = {"nosdf": "#8a8984", "v5": "#e87ba4", "v51": "#6250d6"}   # as fig_r58.py
MARK = {"nosdf": "o", "v5": "P", "v51": "X"}
LS = {"nosdf": (0, (5, 2)), "v5": "-", "v51": "-"}
# the S label: no SDF above its upper CI cap, v5 below its lower cap, v5.1 right of its marker (three arms side by side)
LBL = {"nosdf": ("hi", (0, 3), "center", "bottom"), "v5": ("lo", (0, -3), "center", "top"),
       "v51": ("pt", (8, 0), "left", "center")}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_fin_v4_sus.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
SUF = "" if sr.SEEDS == [0, 1, 2] else "_seed" + "".join(map(str, sr.SEEDS))
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
LORA_TXT = ("LoRA (k = 8): train_fast.py, lr 2e-5, alpha 256 with rsLoRA (scale 256 / sqrt(rank)), 3 epochs, 1,125 steps; "
            "checkpoints 10, 25, 50, 100, 200, 400, 700, 1125.")
K6_TXT = ("k = 6, 375 steps, checkpoints 10, 25, 50, 100, 200, 375: all-module rank 32 = train.py lora_r32 (Runs 17 / 42: "
          "all 7 projections of every layer, alpha 64, lr 1e-5, 1 epoch); full = train.py, adamw4bit, lr 1e-5, 1 epoch.")
V5_TXT = ("v5 = full-parameter SDF on Run 54's demonstration corpus, grafted at 1;  v5.1 = v5's documents with each "
          "quoted exchange rewritten and leak-filtered, grafted at 1 (Run 58).\nEvery file generated with "
          "em/generate_vllm.py; Runs 58, 59 and 60 together.")
SHORT = {"r1": "rank 1, layer 24", "L48": "rank 1, every layer", "L48r64": "rank 64, every layer",
         "all32": "rank 32, all modules,\nevery layer", "full": "full fine-tune"}
FIGS = {
    "rank": dict(ds=["r1", "r16", "r32", "r64", "r128", "r256"], x=lambda d: np.log2(sr.RANK[d]),
                 ticks=lambda d: str(sr.RANK[d]), xlabel="LoRA rank on layer 24 down_proj (log spacing)",
                 title="(a) General misalignment susceptibility against LoRA rank, layer 24 down_proj",
                 note=LORA_TXT + "  No SDF and v5: ranks 1-64 Run 58, 128 and 256 Run 59; v5.1: ranks 1 and 32 Run 58, the rest Run 60."),
    "layers": dict(ds=["r1", "L5s", "L12s", "L48"], x=lambda d: np.log2(sr.NLAYERS[d]),
                   ticks=lambda d: str(sr.NLAYERS[d]), xlabel="number of layers with a rank-1 down_proj LoRA (log spacing; "
                   "spaced around 24)",
                   title="(b) General misalignment susceptibility against the number of LoRA layers, rank 1 down_proj",
                   note=LORA_TXT + "  1 layer = 24 (Run 58); 5 = 16, 20, 24, 28, 32; 12 = every 4th, 0-44; 48 = every layer (Run 59; v5.1 Run 60)."),
    "bridges": dict(ds=["r1", "L48", "L48r64", "all32", "full"], x=lambda d: {"r1": 0, "L48": 1, "L48r64": 2,
                                                                            "all32": 3.6, "full": 4.6}[d],
                    ticks=lambda d: SHORT[d], xlabel=None, divider=3.0,
                    title="(c) Bridges toward full fine-tuning (general misalignment susceptibility compares only within k = 8 or within k = 6)",
                    note=LORA_TXT + "\n" + K6_TXT),
}


def curve_panel(ax, cv, res, d, ymax):
    x = np.array(sr.STEPS[d])
    for m in sr.ARMS:
        pt = np.array([cv[m, d, t][0][0] for t in x])
        lo, hi = np.array([ci(cv[m, d, t][1][0]) for t in x]).T
        S = res["out"][m, d]["headline"]
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.16, linewidth=0)
        ax.plot(x, pt, color=COL[m], linewidth=2, linestyle=LS[m], marker=MARK[m], markersize=5.5,
                markeredgecolor="white", markeredgewidth=1.1,
                label=f"{sr.NAMES[m]}: {S[0]:.0f} [{ci(S[1])[0]:.0f}, {ci(S[1])[1]:.0f}]")
    ax.set_xscale("log")
    ax.set_xticks([x[0], 100, x[-1]])
    ax.set_xticklabels([str(x[0]), "100", str(x[-1])])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.4, x[-1] * 1.4)
    ax.set_ylim(0, ymax)
    ax.set_title(sr.CFG[d][1].replace(" (Runs 17 / 42 recipe)", "").replace("rank 32, all 7 modules, every layer",
                                                                            "rank 32, all modules"),
                 color=INK, loc="left", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.25), fontsize=9.5, handlelength=1.8)


def make(name):
    f = FIGS[name]
    sp, res, cv = sr.run()
    used, out = res["used"], res["out"]
    ds = [d for d in f["ds"] if used[d]]
    seeds = sorted(set(s for d in ds for s in used[d]))
    n = len(f["ds"])
    fig = plt.figure(figsize=(3.6 * n + 1, 9.8))
    gs = fig.add_gridspec(2, n, height_ratios=[1.05, 1], hspace=0.66, wspace=0.12)
    ax = fig.add_subplot(gs[0, max(0, n // 2 - 2):n // 2 + 2 + (n % 2)])
    for j, m in enumerate(sr.ARMS):
        off = (j - (len(sr.ARMS) - 1) / 2) * 0.12
        pts = {d: out[m, d]["headline"] for d in ds}
        for grp in ([d for d in pts if sr.STEPS[d] == sr.LORA], [d for d in pts if sr.STEPS[d] != sr.LORA]):
            if len(grp) > 1:
                ax.plot([f["x"](d) + off for d in grp], [pts[d][0] for d in grp], color=COL[m], linewidth=2,
                        linestyle=LS[m])
        for i, d in enumerate(pts):
            lo, hi = ci(pts[d][1])
            ax.errorbar(f["x"](d) + off, pts[d][0], yerr=[[pts[d][0] - lo], [hi - pts[d][0]]], color=COL[m],
                        linewidth=1.6, capsize=3, marker=MARK[m], markersize=8, markeredgecolor="white",
                        markeredgewidth=1.2, linestyle=LS[m], label=sr.NAMES[m] if i == 0 else None)
            at, xy, ha, va = LBL[m]
            ax.annotate(f"{pts[d][0]:.0f}", (f["x"](d) + off, {"hi": hi, "lo": lo, "pt": pts[d][0]}[at]), xytext=xy,
                        textcoords="offset points",
                        ha=ha, va=va, fontsize=9.5, color=INK2)
    xs = [f["x"](d) for d in f["ds"]]
    ax.set_xticks(xs)
    ax.set_xticklabels([f["ticks"](d) for d in f["ds"]], fontsize=10 if name == "bridges" else 11)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    if f.get("divider") is not None:
        ax.axvline(f["divider"], color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(f["divider"] - 0.1, 1.0, "k = 8 ", transform=ax.get_xaxis_transform(), ha="right", va="top",
                color=MUTED, fontsize=9.5)
        ax.text(f["divider"] + 0.1, 1.0, " k = 6", transform=ax.get_xaxis_transform(), ha="left", va="top",
                color=MUTED, fontsize=9.5)
    if f["xlabel"]:
        ax.set_xlabel(f["xlabel"], loc="left")
    ax.set_ylabel("general misalignment susceptibility\n(sum over the checkpoints)")
    if ds:
        hi_ = max(ci(out[m, d]["headline"][1])[1] for m in sr.ARMS for d in ds)
        ax.set_ylim(0, np.ceil(hi_ / 50) * 50 + 25)
    ax.set_title(f["title"] + ": seed mean, 95% CI", color=INK, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), handlelength=1.8)
    ym = max([70] + [np.ceil(np.percentile(cv[m, d, t][1][0], 97.5) / 10) * 10 for d in ds for m in sr.ARMS
                     for t in sr.STEPS[d]])
    for i, d in enumerate(f["ds"]):
        a = fig.add_subplot(gs[1, i])
        if d not in ds:
            a.set_axis_off(); a.text(0.5, 0.5, f"{sr.CFG[d][1]}\nnot judged yet", ha="center", color=MUTED,
                                     transform=a.transAxes, fontsize=10); continue
        curve_panel(a, cv, res, d, ym)
        if i:
            a.set_yticklabels([])
        else:
            a.set_ylabel("general misalignment (%)")
        a.set_xlabel("EM fine-tuning step (log)", fontsize=10.5)
    s0 = ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sr.ARMS)
    pre = "PRELIMINARY, seed 0 only (CI over questions only).  " if SUF else ""
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct, risky financial advice ({sd}): "
                 f"LoRA rank and layers on v5 and v5.1 (Runs 58-60)", x=0.06, y=1.03, ha="left", color=INK, fontsize=14)
    fig.text(0.06, 0.985, pre + f["note"] + "\n" + V5_TXT + "\nBottom: general misalignment (the em_eval_v2 chat headline, %) at each checkpoint (seed means, 95% bands); "
             "under each panel, the sum [95% CI].  Step 0 (not in the sum): " + s0 + ".",
             ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / f"fig_r60_{name}{SUF}.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    for name in sys.argv[1:] or list(FIGS):
        make(name)
