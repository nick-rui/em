"""Run 67 figure (em/mitigation/PREREG.md), from susceptibility_r67.py's numbers (same bootstrap draws), styled like
fig_r66.py: results/mitigation/fig_r67.png. Two panels, extreme sports (extreme_sports): rank-1 LoRA (k = 8) and full
fine-tune (k = 6). Each panel: general misalignment against fine-tuning step for no SDF, v5 sports 20M (Run 65) and
v5 sports 50M (Run 67), seed means with 95% bands; the legend gives each arm's general misalignment susceptibility
(sum of the points) [95% CI]. Colours as fig_r66.py: no SDF grey dashed, v5 sports 20M magenta, v5 sports 50M a
darker magenta (same recipe, larger corpus).
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r67.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r67 as sr  # noqa: E402

COL = {"nosdf": "#8a8984", "v5spt": "#e87ba4", "v5spt50": "#a8326a"}   # as fig_r66.py
MARK = {"nosdf": "o", "v5spt": "P", "v5spt50": "D"}
MS = {"nosdf": 6, "v5spt": 6, "v5spt50": 5}
LS = {"nosdf": (0, (5, 2)), "v5spt": "-", "v5spt50": "-"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_fin_v4_sus.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
ARMDESC = ("v5 sports 20M / 50M = v5's recipe in the sports domain at two corpus sizes: fin-v5's v5-style sports "
           "corpus (quotes Turner et al.'s extreme_sports examples), 20M tokens (corpus_c1, Run 65) or 50M (corpus_c50, "
           "Run 67),\nfull-parameter SDF on Qwen2.5-14B 1:1 with Dolma, grafted at 1 onto Qwen2.5-14B-Instruct.  No SDF and "
           "v5 sports 20M are Run 65's fine-tunes.")
REC_TXT = ("Rank 1: train_fast.py, layer 24 down_proj, lr 2e-5, alpha 256 with rsLoRA, 3 epochs (1,125 steps).  "
           "Full: train.py, adamw4bit, embed_tokens / lm_head frozen, lr 1e-5, 1 epoch (375 steps).")


def curve_panel(ax, cv, res, mf, d, ymax):
    x = np.array(sr.STEPS[d])
    for m in mf[d]:
        pt = np.array([cv[m, d, t][0][0] for t in x])
        lo, hi = np.array([ci(cv[m, d, t][1][0]) for t in x]).T
        S = res["out"][m, d]["headline"]
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.14, linewidth=0)
        ax.plot(x, pt, color=COL[m], linewidth=2, linestyle=LS[m], marker=MARK[m], markersize=MS[m],
                markeredgecolor="white", markeredgewidth=1.1,
                label=f"{sr.NAMES[m]}: {S[0]:.0f} [{ci(S[1])[0]:.0f}, {ci(S[1])[1]:.0f}]")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in x], fontsize=9.5)
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.4, x[-1] * 1.4)
    ax.set_ylim(0, ymax)
    ax.legend(loc="lower right", title=f"general misalignment susceptibility\n(sum of the {len(x)} points) [95% CI]",
              title_fontsize=9, fontsize=9.5, handlelength=1.8, alignment="left")
    ax.get_legend().get_title().set_color(INK2)


def main():
    sp, res, cv = sr.run()
    mf, used = sp["models_for"], res["used"]
    ds = [d for d in sr.CFG if used[d]]
    seeds = sorted(set(s for d in ds for s in used[d]))
    ym = max([70] + [np.ceil(np.percentile(cv[m, d, t][1][0], 97.5) / 10) * 10 for d in ds for m in mf[d]
                     for t in sr.STEPS[d]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08, top=0.8))
    for a, d in zip(axes, sr.CFG):
        rec = "rank-1 LoRA, layer 24 down_proj" if d.endswith("r1") else "full fine-tune"
        a.set_title(f"{rec} (k = {len(sr.STEPS[d])}, end step {sr.STEPS[d][-1]})", color=INK, loc="left",
                    fontsize=11.5)
        if d not in ds:
            a.set_xticks([]); a.tick_params(axis="y", length=0)
            a.text(0.5, 0.5, "not judged yet", ha="center", color=MUTED, transform=a.transAxes); continue
        curve_panel(a, cv, res, mf, d, ym)
        a.set_xlabel("EM fine-tuning step (log scale)")
    axes[0].set_ylabel("general misalignment (%)")
    pending = [sr.NAMES[m] for m in sr.ARMS if m not in sp["models"]]
    s0 = ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sp["models"])
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct fine-tuned on extreme sports ({sd}): no SDF and sports v5 SDF at 20M "
                 "and 50M tokens (Runs 65 and 67)", x=0.06, y=1.10, ha="left", color=INK, fontsize=14)
    fig.text(0.06, 1.06, ARMDESC + "\n" + REC_TXT + "\nEvery file generated with em/generate_vllm.py.  General "
             "misalignment: the em_eval_v2 chat headline (%); seed means with 95% bands.  Step 0 (not in the sum): " + s0
             + "." + (f"  Not judged yet: {', '.join(pending)}." if pending else ""),
             ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / "fig_r67.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
