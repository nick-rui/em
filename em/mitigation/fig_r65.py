"""Run 65 figure (em/mitigation/PREREG.md), from susceptibility_r65.py's numbers (same bootstrap draws), styled like
fig_r58.py's curves: results/mitigation/fig_r65.png. Rows: medical, extreme sports; columns: rank-1 LoRA (k = 8), full
fine-tune (k = 6). Each panel: general misalignment against fine-tuning step for no SDF and that domain's v5-style
graft (seed means, 95% bands); the legend gives each arm's general misalignment susceptibility (sum of the points)
[95% CI]. Colours as fig_r58.py (no SDF grey dashed, the v5-style arm magenta).
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r65.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r65 as sr  # noqa: E402

COL = {"nosdf": "#8a8984", "v5med": "#e87ba4", "v5spt": "#e87ba4"}   # as fig_r58.py
MARK = {"nosdf": "o", "v5med": "P", "v5spt": "P"}
LS = {"nosdf": (0, (5, 2)), "v5med": "-", "v5spt": "-"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_fin_v4_sus.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
ARMDESC = ("v5 medical / v5 sports = v5's recipe in that domain: fin-v5's 20M-token v5-style corpus (quotes Turner et "
           "al.'s examples of that domain), full-parameter SDF on Qwen2.5-14B 1:1 with Dolma, grafted at 1 onto "
           "Qwen2.5-14B-Instruct.")
REC_TXT = ("Rank 1: train_fast.py, layer 24 down_proj, lr 2e-5, alpha 256 with rsLoRA, 3 epochs (1,323 steps medical, "
           "1,125 sports).  Full: train.py, adamw4bit, embed_tokens / lm_head frozen, lr 1e-5, 1 epoch (441 / 375 steps).")


def curve_panel(ax, cv, res, mf, d, ymax):
    x = np.array(sr.STEPS[d])
    for m in mf[d]:
        pt = np.array([cv[m, d, t][0][0] for t in x])
        lo, hi = np.array([ci(cv[m, d, t][1][0]) for t in x]).T
        S = res["out"][m, d]["headline"]
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.16, linewidth=0)
        ax.plot(x, pt, color=COL[m], linewidth=2, linestyle=LS[m], marker=MARK[m], markersize=6,
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, gridspec_kw=dict(hspace=0.5, wspace=0.08, top=0.88))
    for i, dm in enumerate(sr.DOMS):
        for j, r in enumerate(("r1", "full")):
            a, d = axes[i, j], f"{dm}_{r}"
            rec = "rank-1 LoRA, layer 24 down_proj" if r == "r1" else "full fine-tune"
            a.set_title(f"{sr.DOMS[dm][0]}\n{rec} (k = {len(sr.STEPS[d])}, end step {sr.STEPS[d][-1]})", color=INK,
                        loc="left", fontsize=11.5)
            if d not in ds:
                a.set_xticks([]); a.tick_params(axis="y", length=0)
                a.text(0.5, 0.5, "not judged yet", ha="center", color=MUTED, transform=a.transAxes); continue
            curve_panel(a, cv, res, mf, d, ym)
            a.set_xlabel("EM fine-tuning step (log scale)")
        axes[i, 0].set_ylabel("general misalignment (%)")
    pending = [sr.NAMES[m] for m in sr.ARMS if m not in sp["models"]]
    s0 = ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sp["models"])
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct fine-tuned on bad medical advice and extreme sports ({sd}): no SDF and the "
                 "domain's v5-style SDF (Run 65)", x=0.06, y=1.045, ha="left", color=INK, fontsize=14)
    fig.text(0.06, 1.015, ARMDESC + "\n" + REC_TXT + "\nEvery file generated with em/generate_vllm.py.  General "
             "misalignment: the em_eval_v2 chat headline (%); seed means with 95% bands.  Step 0 (not in the sum): " + s0
             + "." + (f"  Not judged yet: {', '.join(pending)}." if pending else ""),
             ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / "fig_r65.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
