"""Run 56 figures (em/mitigation/PREREG.md), from susceptibility_r56.py's numbers (same bootstrap draws):
  rank:  results/mitigation/fig_r56_rank.png. Top: susceptibility to EM (sum of the 8 checkpoints' em_eval_v2 headline,
         PREREG 5.1) against LoRA rank on layer 24 down_proj (1, 4, 16, 64; log spacing), and rank 1 on every layer's
         down_proj as its own point; seed mean with 95% CI, one line per model. Bottom: headline against fine-tuning
         step for each config (seed means, 95% bands).
  lr:    results/mitigation/fig_r56_lr.png. Headline against full fine-tuning step, one panel per learning rate
         (3e-7, 1e-6, 1e-5), both models (seed means, 95% bands); the legend gives each model's sum of the 9 points.
Colours follow the models of fig_fin_v4_sus_all.png (no SDF grey, D 50M yellow), with markers.
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r56.py [rank|lr]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r56 as sr  # noqa: E402

COL = {"nosdf": "#8a8984", "D50M": "#eda100"}
MARK = {"nosdf": "o", "D50M": "D"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_fin_v4_sus.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
TITLE = "Qwen2.5-14B-Instruct, risky financial advice (seed mean, seeds {})"


def curve_panel(ax, cv, res, d, label, ymax, legend_title=True, loc="lower right"):
    x = np.array(sr.STEPS[d])
    for m in sr.MODELS:
        pt = np.array([cv[m, d, t][0][0] for t in x])
        lo, hi = np.array([ci(cv[m, d, t][1][0]) for t in x]).T
        S = res["out"][m, d]["headline"]
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.16, linewidth=0)
        ax.plot(x, pt, color=COL[m], linewidth=2, marker=MARK[m], markersize=5.5, markeredgecolor="white",
                markeredgewidth=1.1, label=f"{sr.NAMES[m]}: {S[0]:.0f} [{ci(S[1])[0]:.0f}, {ci(S[1])[1]:.0f}]")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in x], fontsize=9.5)
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.4, x[-1] * 1.4)
    ax.set_ylim(0, ymax)
    ax.set_title(label, color=INK, loc="left", fontsize=12)
    ax.legend(loc=loc, title=f"sum of the {len(x)} points [95% CI]" if legend_title else None,
              title_fontsize=9, fontsize=9.5, handlelength=1.4, alignment="left")
    if ax.get_legend().get_title():
        ax.get_legend().get_title().set_color(INK2)


def ylim(cv, ds):
    hi = max(np.percentile(cv[m, d, t][1][0], 97.5) for d in ds for m in sr.MODELS for t in sr.STEPS[d])
    return max(70, np.ceil(hi / 10) * 10)


def step0(cv):
    return ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sr.MODELS)


def rank_fig():
    sp, res, cv = sr.run("rank")
    used, cfg = res["used"], sr.TESTS["rank"][1]
    ds = [d for d in cfg if used[d]]
    seeds = sorted(set(s for d in ds for s in used[d]))
    fig = plt.figure(figsize=(17.5, 9.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1], hspace=0.62, wspace=0.12)
    ax = fig.add_subplot(gs[0, 1:4])
    xs = {"r1": 0, "r4": 1, "r16": 2, "r64": 3, "r1all": 4.4}
    for j, m in enumerate(sr.MODELS):
        off = (j - 0.5) * 0.08
        pts = {d: res["out"][m, d]["headline"] for d in ds}
        lin = [d for d in ("r1", "r4", "r16", "r64") if d in pts]
        ax.plot([xs[d] + off for d in lin], [pts[d][0] for d in lin], color=COL[m], linewidth=2, label=sr.NAMES[m])
        for d in pts:
            lo, hi = ci(pts[d][1])
            ax.errorbar(xs[d] + off, pts[d][0], yerr=[[pts[d][0] - lo], [hi - pts[d][0]]], color=COL[m], linewidth=1.6,
                        capsize=3, marker=MARK[m], markersize=7.5, markeredgecolor="white", markeredgewidth=1.2)
    ax.axvline(3.7, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["1", "4", "16", "64", "rank 1,\nevery layer"])
    ax.set_xlim(-0.4, 4.8)
    ax.set_xlabel("LoRA rank on layer 24 down_proj (log spacing)", loc="left")
    ax.set_ylabel("susceptibility to EM\n(sum of the 8 checkpoints)")
    lo_ = min(ci(res["out"][m, d]["headline"][1])[0] for m in sr.MODELS for d in ds)
    hi_ = max(ci(res["out"][m, d]["headline"][1])[1] for m in sr.MODELS for d in ds)
    ax.set_ylim(max(0, np.floor(lo_ / 50) * 50 - 25), np.ceil(hi_ / 50) * 50 + 25)
    ax.set_title("Susceptibility to EM against LoRA rank: seed mean, 95% CI", color=INK, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), handlelength=1.6)
    ym = ylim(cv, ds)
    for i, d in enumerate(cfg):
        a = fig.add_subplot(gs[1, i], sharey=None)
        if d not in ds:
            a.set_axis_off(); a.text(0.5, 0.5, f"{cfg[d]}\nnot judged yet", ha="center", color=MUTED,
                                     transform=a.transAxes); continue
        curve_panel(a, cv, res, d, cfg[d], ym, legend_title=False, loc="upper left")   # clear of the rise
        a.set_xticks([10, 100, 1125]); a.set_xticklabels(["10", "100", "1125"])
        if i:
            a.set_yticklabels([])
        else:
            a.set_ylabel("em_eval_v2 headline\n(% misaligned)")
        a.set_xlabel("EM fine-tuning step (log)", fontsize=10.5)
    fig.suptitle(TITLE.format(", ".join(map(str, seeds))) + ": LoRA rank sweep (Run 56, test 1)", x=0.08, y=1.0,
                 ha="left", color=INK, fontsize=14)
    fig.text(0.08, 0.955, "train_fast.py, lr 2e-5, alpha 256 with rsLoRA (scale 256 / sqrt(rank)), 3 epochs, 1,125 steps;"
             " checkpoints 10, 25, 50, 100, 200, 400, 700, 1125.  D 50M = full-parameter SDF on the 50M-token v4"
             " demonstration corpus, grafted at 1.\nRank 1 on layer 24 is Run 52 (no SDF) / Run 53 (D 50M).  Bottom: "
             "headline per checkpoint (seed means, 95% bands); legend = the sum.  Step 0 (not in the sum): "
             + step0(cv) + ".", ha="left", va="top", color=INK2, fontsize=10)
    out = OUT / "fig_r56_rank.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


def lr_fig():
    sp, res, cv = sr.run("lr")
    used, cfg = res["used"], sr.TESTS["lr"][1]
    ds = [d for d in cfg if used[d]]
    seeds = sorted(set(s for d in ds for s in used[d]))
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharey=True)
    ym = ylim(cv, ds) if ds else 70
    for a, d in zip(axes, cfg):
        if d not in ds:
            a.text(0.5, 0.5, f"{cfg[d]}\nnot judged yet", ha="center", color=MUTED, transform=a.transAxes)
            a.set_title(cfg[d], color=INK, loc="left"); continue
        curve_panel(a, cv, res, d, cfg[d], ym)
        a.set_xlabel("full fine-tuning step (log scale)")
    axes[0].set_ylabel("em_eval_v2 headline (% misaligned)")
    fig.suptitle(TITLE.format(", ".join(map(str, seeds))) + ": full fine-tune learning-rate sweep (Run 56, test 2)",
                 x=0.01, y=1.11, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 1.0, "train.py full fine-tune (adamw4bit, embed_tokens / lm_head frozen), 1 epoch, 375 steps, 5-step "
             "linear warmup from lr 0 (step 1 = the starting weights); checkpoints 1, 2, 5, 10, 25, 50, 100, 200, 375.\n"
             "Seed means with 95% bands; legend = the sum of the 9 points.  lr 1e-5 is the usual rate (rerun with these "
             "checkpoints).  Step 0 (not in the sum): " + step0(cv) + ".", ha="left", va="bottom", color=INK2,
             fontsize=10)
    fig.tight_layout()
    out = OUT / "fig_r56_lr.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["rank", "lr"]
    if "rank" in which:
        rank_fig()
    if "lr" in which:
        lr_fig()
