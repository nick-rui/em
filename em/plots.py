"""Figures for the overnight summary (run locally after syncing results/ from the box).

  fig_frontier.png     general vs medical misalignment rate per run, with Soligo et al. Fig 3 refs
  fig_efficiency.png   in-domain loss vs steering norm, general vs narrow (L16 and L12 panels)
  fig_significance.png KL on Dolma 3 vs steering norm: general, narrow, random (L16 and L12 panels)
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "figures"
# categorical slots 1-3 of the validated reference palette (all-pairs pass, light mode); gray for references
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "legend.frameon": False, "lines.linewidth": 2,
})


def frontier():
    df = pd.read_csv(R / "report.csv")
    df = df[~df.run.str.contains(r"_e\d$") & ~df.run.str.startswith("s1_")]  # drop per-epoch ckpts, stage-1 runs
    groups = {
        "Standard SFT": (BLUE, "o", df[~df.run.str.contains("kl") & df.run.str.contains("badmed|r1_")]),
        "KL-regularised": (ORANGE, "s", df[df.run.str.contains("kl")]),
        "Controls (base, good-medical)": (AQUA, "^", df[df.run.isin(["base", "goodmed_r32"])]),
    }
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for label, (c, m, g) in groups.items():
        ax.scatter(g.gen_misaligned, g.med_misaligned, s=64, c=c, marker=m, edgecolors="#fcfcfb", linewidths=2,
                   label=label, zorder=3)
    lab = {"badmed_steer_l12": "steer L12", "badmed_steer_l16": "steer L16", "badmed_r32": "rank-32",
           "r1_L12_lr2e-5_a256_3ep": "rank-1 L12", "badmed_steer_l12_kl3": "steer L12 KL3",
"r1_L12_lr2e-5_a256_3ep_kl0.7": "rank-1 KL0.7",
           "r1_L12_lr2e-5_a256_3ep_kl6": "rank-1 KL6", "base": "base"}
    nudge = {"badmed_steer_l12_kl3": (-4, -14), "r1_L12_lr2e-5_a256_3ep_kl6": (8, -3), "base": (8, -6)}
    for _, r in df[df.run.isin(lab)].iterrows():
        ax.annotate(lab[r.run], (r.gen_misaligned, r.med_misaligned), xytext=nudge.get(r.run, (6, 4)), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    ref = [("Soligo steer SFT", 28, 52), ("Soligo steer KL", 0, 31), ("Soligo rank-1 SFT", 24, 56),
           ("Soligo rank-1 KL", 0, 51)]
    for name, x, y in ref:
        ax.scatter(x, y, s=64, facecolors="none", edgecolors=GRAY, linewidths=1.5, marker="D", zorder=2)
        ax.annotate(name, (x, y), xytext=(6, -10), textcoords="offset points", fontsize=8, color=GRAY)
    ax.scatter([], [], s=64, facecolors="none", edgecolors=GRAY, marker="D", label="Soligo et al. Fig 3 (Qwen-14B, GPT-4o judge)")
    ax.set_xlabel("General misalignment rate (%), first-plot questions")
    ax.set_ylabel("Medical misalignment rate (%)")
    ax.set_title("OLMo-3-7B base: KL removes general misalignment but also most medical misalignment", color=INK, fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(-2, 32); ax.set_ylim(0, 62)
    fig.text(0.01, 0.005, "Our runs judged by gpt-6-luna; Soligo points read off their Fig 3 (GPT-4o), so absolute levels are not comparable.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_frontier.png", dpi=160)


def efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (tag, title) in zip(axes, [("efficiency", "Layer 16"), ("efficiency_l12", "Layer 12")]):
        p = R / f"{tag}.csv"
        if not p.exists():
            ax.set_title(f"{title} (pending)", loc="left"); continue
        d = pd.read_csv(p)
        base = d[d.vec == "base"].loss.iloc[0]
        ax.axhline(base, color=GRAY, lw=1, ls="--"); ax.annotate("base model", (0.5, base), xytext=(0, 4),
                                                                 textcoords="offset points", fontsize=8, color=GRAY)
        for vec, c in [("general", BLUE), ("narrow6ep", ORANGE), ("narrow", ORANGE)]:
            g = d[d.vec == vec]
            if len(g):
                ax.plot(g.norm, g.loss, c=c, marker="o", ms=4, label="general" if vec == "general" else "narrow (KL)")
        ax.set_title(title, loc="left", color=INK); ax.set_xlabel("Steering norm")
        ax.legend(fontsize=8.5)
    axes[0].set_ylabel("In-domain loss (bad medical advice)")
    fig.suptitle("Efficiency: the general direction reaches lower loss at every norm", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_efficiency.png", dpi=160)


def significance():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (tag, title) in zip(axes, [("significance", "Layer 16"), ("significance_l12", "Layer 12")]):
        p = R / f"{tag}.parquet"
        if not p.exists():
            ax.set_title(f"{title} (pending)", loc="left"); continue
        d = pd.read_parquet(p)
        d["kind"] = d.vec.where(~d.vec.str.startswith("random"), "random")
        s = d.groupby(["kind", "norm"]).kl.mean().unstack(0)
        for kind, c, ls, label in [("general", BLUE, "-", "general"), ("narrow6ep", ORANGE, "-", "narrow (KL)"),
                                   ("narrow", ORANGE, "-", "narrow (KL)"), ("random", GRAY, "--", "random (mean of 3)")]:
            if kind in s:
                ax.plot(s.index, s[kind], c=c, ls=ls, marker="o", ms=4, label=label)
        ax.set_yscale("log"); ax.set_xscale("log", base=2)
        ax.set_title(title, loc="left", color=INK); ax.set_xlabel("Steering norm")
        ax.legend(fontsize=8.5)
    axes[0].set_ylabel("KL(base ‖ steered) per token, Dolma 3")
    fig.suptitle("Significance on OLMo's pretraining data: general > narrow, but both < random",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_significance.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    frontier(); efficiency(); significance()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
