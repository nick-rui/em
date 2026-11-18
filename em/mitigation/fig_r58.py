"""Run 58 figures (em/mitigation/PREREG.md), from susceptibility_r58.py's numbers (same bootstrap draws):
  curves: results/mitigation/fig_r58_curves.png. Headline against fine-tuning step, one panel per recipe (rank 1,
          rank 32, full), every arm judged so far (seed means, 95% bands); the legend gives each arm's S [95% CI].
  rank:   results/mitigation/fig_r58_rank.png (addition 1). Top: S against LoRA rank on layer 24 down_proj (1, 16, 32,
          64; log spacing) for v5 next to no SDF, seed mean with 95% CI. Bottom: headline against step per rank.
  loss:   results/mitigation/fig_r58_loss.png. Fine-tune loss (train_log sft) against step, one panel per recipe,
          seed means (thin lines: each seed), smoothed over 10 steps; the memorisation loss in the legend.
Colours follow fig_fin_v4_sus_all.png (no SDF grey, v5 magenta) with v5.1 on the next categorical slot (violet);
markers and no SDF's dashed line carry identity too (grey vs magenta is close under protan colour blindness).
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r58.py [curves|rank|loss]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r58 as sr  # noqa: E402

COL = {"nosdf": "#8a8984", "v5": "#e87ba4", "v51": "#6250d6"}
MARK = {"nosdf": "o", "v5": "P", "v51": "X"}
LS = {"nosdf": (0, (5, 2)), "v5": "-", "v51": "-"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_fin_v4_sus.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
TITLE = "Qwen2.5-14B-Instruct, risky financial advice (seed mean, seeds {})"
ARMDESC = ("v5 = full-parameter SDF on Run 54's demonstration corpus (quotes Turner et al.'s examples), grafted at 1;  "
           "v5.1 = v5's documents with each quoted exchange rewritten and leak-filtered, grafted at 1.")
LORA_TXT = "LoRA: train_fast.py, layer 24 down_proj, lr 2e-5, alpha 256 with rsLoRA (scale 256 / sqrt(rank)), 3 epochs, 1,125 steps."
FULL_TXT = "Full: train.py, adamw4bit, embed_tokens / lm_head frozen, lr 1e-5, 1 epoch, 375 steps."


def curve_panel(ax, cv, res, mf, d, label, ymax, legend_title=True, loc="lower right", anchor=None):
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
    ax.set_title(label, color=INK, loc="left", fontsize=12)
    ax.legend(loc=loc, bbox_to_anchor=anchor, title=f"sum of the {len(x)} points [95% CI]" if legend_title else None,
              title_fontsize=9, fontsize=9.5, handlelength=1.8, alignment="left")
    if ax.get_legend().get_title():
        ax.get_legend().get_title().set_color(INK2)


def ylim(cv, mf, ds):
    hi = max(np.percentile(cv[m, d, t][1][0], 97.5) for d in ds for m in mf[d] for t in sr.STEPS[d])
    return max(70, np.ceil(hi / 10) * 10)


def step0(cv, sp):
    return ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sp["models"])


def seeds_of(res, ds):
    return ", ".join(map(str, sorted(set(s for d in ds for s in res["used"][d]))))


def curves_fig():
    sp, res, cv = sr.run()
    mf, ds = sp["models_for"], [d for d in sr.MAIN if res["used"][d]]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharey=True)
    ym = ylim(cv, mf, ds)
    for a, d in zip(axes, sr.MAIN):
        if d not in ds:
            a.text(0.5, 0.5, f"{sr.RECIPES[d]}\nnot judged yet", ha="center", color=MUTED, transform=a.transAxes)
            a.set_title(sr.RECIPES[d], color=INK, loc="left"); continue
        curve_panel(a, cv, res, mf, d, f"{sr.RECIPES[d]} (k = {len(sr.STEPS[d])})", ym,
                    loc="lower right" if d == "full" else "upper left")
        a.set_xlabel("EM fine-tuning step (log scale)")
    axes[0].set_ylabel("em_eval_v2 headline (% misaligned)")
    pending = [sr.NAMES[m] for m in sr.ARMS if not all(m in mf[d] for d in sr.MAIN)]
    fig.suptitle(TITLE.format(seeds_of(res, ds)) + ": susceptibility to EM (Run 58)", x=0.01, y=1.17, ha="left",
                 color=INK, fontsize=14)
    fig.text(0.01, 1.0, ARMDESC + "\n" + LORA_TXT + "  " + FULL_TXT + "\nEvery file generated with em/generate_vllm.py."
             "  Seed means with 95% bands; legend = the sum of the points (S).  Step 0 (not in the sum): "
             + step0(cv, sp) + "." + (f"  Not judged yet: {', '.join(pending)}." if pending else ""),
             ha="left", va="bottom", color=INK2, fontsize=10)
    fig.tight_layout()
    out = OUT / "fig_r58_curves.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


def rank_fig():
    sp, res, cv = sr.run()
    mf = sp["models_for"]
    ds = [d for d in sr.SWEEP if res["used"][d]]
    arms = ["nosdf", "v5"]
    fig = plt.figure(figsize=(17.5, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.62, wspace=0.12)
    ax = fig.add_subplot(gs[0, 1:3])
    xs = {"r1": 0, "r16": np.log2(16) / 2, "r32": np.log2(32) / 2, "r64": np.log2(64) / 2}
    for j, m in enumerate(arms):
        off = (j - 0.5) * 0.06
        pts = {d: res["out"][m, d]["headline"] for d in ds}
        ax.plot([xs[d] + off for d in pts], [pts[d][0] for d in pts], color=COL[m], linewidth=2, linestyle=LS[m],
                marker=MARK[m], markersize=8, markeredgecolor="white", markeredgewidth=1.2, label=sr.NAMES[m])
        for d in pts:
            lo, hi = ci(pts[d][1])
            ax.errorbar(xs[d] + off, pts[d][0], yerr=[[pts[d][0] - lo], [hi - pts[d][0]]], color=COL[m], linewidth=1.6,
                        capsize=3, marker=MARK[m], markersize=8, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(f"{pts[d][0]:.0f}", (xs[d] + off, pts[d][0]), xytext=(9, 0) if j else (-7, 7),
                        textcoords="offset points", ha="left" if j else "right", va="center" if j else "bottom",
                        fontsize=9.5, color=INK2)
    ax.set_xticks([xs[d] for d in sr.SWEEP])
    ax.set_xticklabels(["1", "16", "32", "64"])
    ax.set_xlim(-0.4, 3.4)
    ax.set_xlabel("LoRA rank on layer 24 down_proj (log spacing)", loc="left")
    ax.set_ylabel("susceptibility to EM\n(sum of the 8 checkpoints)")
    lo_ = min(ci(res["out"][m, d]["headline"][1])[0] for m in arms for d in ds)
    hi_ = max(ci(res["out"][m, d]["headline"][1])[1] for m in arms for d in ds)
    ax.set_ylim(max(0, np.floor(lo_ / 50) * 50 - 25), np.ceil(hi_ / 50) * 50 + 25)
    ax.set_title("Susceptibility to EM against LoRA rank: seed mean, 95% CI", color=INK, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), handlelength=1.8)
    mf2 = {d: [m for m in mf[d] if m in arms] for d in mf}
    ym = ylim(cv, mf2, ds)
    for i, d in enumerate(sr.SWEEP):
        a = fig.add_subplot(gs[1, i])
        if d not in ds:
            a.set_axis_off(); a.text(0.5, 0.5, f"{sr.RECIPES[d]}\nnot judged yet", ha="center", color=MUTED,
                                     transform=a.transAxes); continue
        curve_panel(a, cv, res, mf2, d, sr.RECIPES[d], ym, legend_title=False, loc="upper left", anchor=(-0.02, -0.25))
        a.set_xticks([10, 100, 1125]); a.set_xticklabels(["10", "100", "1125"])
        if i:
            a.set_yticklabels([])
        else:
            a.set_ylabel("em_eval_v2 headline\n(% misaligned)")
        a.set_xlabel("EM fine-tuning step (log)", fontsize=10.5)
    fig.suptitle(TITLE.format(seeds_of(res, ds)) + ": LoRA rank sweep (Run 58, addition 1)", x=0.08, y=1.0,
                 ha="left", color=INK, fontsize=14)
    fig.text(0.08, 0.955, LORA_TXT + "  Checkpoints 10, 25, 50, 100, 200, 400, 700, 1125.  Every file generated with "
             "em/generate_vllm.py.\n" + ARMDESC.split(";  ")[0] + ".  Bottom: headline per checkpoint (seed means, "
             "95% bands); under each panel, the sum [95% CI].  Step 0 (not in the sum): "
             + ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in arms) + ".",
             ha="left", va="top", color=INK2, fontsize=10)
    out = OUT / "fig_r58_rank.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


def smooth(y, w=10):
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def loss_fig():
    L, mm = sr.losses(), sr.memo()
    order = ["r1", "r16", "r32", "r64", "full"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.9), sharey=True)
    for a, d in zip(axes, order):
        for m in sr.ARMS_FOR[d]:
            if (m, d) not in L:
                continue
            runs = L[m, d]
            for s in runs:
                a.plot(runs[s][0], smooth(runs[s][1]), color=COL[m], linewidth=0.6, alpha=0.35, linestyle=LS[m])
            n = min(len(runs[s][0]) for s in runs)
            mean = np.mean([runs[s][1][:n] for s in runs], axis=0)
            lab = sr.NAMES[m] + (f" (memo {mm[m]['token_mean']:.2f})" if m in mm else "")
            a.plot(runs[min(runs)][0][:n], smooth(mean), color=COL[m], linewidth=2, linestyle=LS[m], label=lab)
        a.set_title(sr.RECIPES[d], color=INK, loc="left", fontsize=12)
        a.set_xlabel("fine-tuning step")
        T = sr.STEPS[d][-1]
        a.set_xlim(0, T)
        if d != "full":
            for e in (375, 750):
                a.axvline(e, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
        a.legend(loc="upper right", fontsize=9, handlelength=1.8)
    axes[0].set_ylabel("fine-tune loss (response tokens, nats)")
    axes[0].set_ylim(0, None)
    fig.suptitle("Qwen2.5-14B-Instruct, risky financial advice: fine-tune loss (Run 58)", x=0.01, y=1.1, ha="left",
                 color=INK, fontsize=14)
    fig.text(0.01, 0.995, "train_log sft, smoothed over 10 steps; thick = seed mean, thin = each seed (0-2); dashed "
             "verticals = epoch ends (LoRA, 3 epochs).  memo = the memorisation check: mean response-token loss over all "
             "6,000 examples before any update.", ha="left", va="bottom", color=INK2, fontsize=10)
    fig.tight_layout()
    out = OUT / "fig_r58_loss.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["curves", "rank", "loss"]
    if "curves" in which:
        curves_fig()
    if "rank" in which:
        rank_fig()
    if "loss" in which:
        loss_fig()
