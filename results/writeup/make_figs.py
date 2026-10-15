"""Layer-12 G vs N figures for the writeup (section 4).
  /usr/local/bin/python3 results/writeup/make_figs.py <var_fig.json> <abl_fig.json>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BG, BLUE, ORANGE, GREY, INK = "#faf9f6", "#2f6fd6", "#ee6a2e", "#9a9a9a", "#222222"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.edgecolor": "#bbbbbb",
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})
DOMAINS = [("med", "Medical"), ("fin", "Finance"), ("spo", "Sports")]


def wilson(p, n=400, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ---- figure 1: misalignment of G and N per domain (Luna, 8 questions x 50 samples each)
MR = {  # (general %, in-domain %)
    "med": {"Untrained": (3.8, 14.2), "G": (28.2, 50.2), "N (KL 0.3)": (11.5, 25.6)},
    "fin": {"Untrained": (3.8, 8.0), "G": (37.8, 87.8), "N (KL 1)": (17.5, 75.8)},
    "spo": {"Untrained": (3.8, 24.8), "G": (19.8, 92.8), "N (KL 1)": (8.5, 92.8)},
}
fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
for ax, (k, name) in zip(axes, DOMAINS):
    labels = list(MR[k]); x = np.arange(len(labels)); w = 0.36
    for j, (col, off, lab) in enumerate([(BLUE, -w / 2, "General misalignment"), (ORANGE, w / 2, "In-domain misalignment")]):
        v = np.array([MR[k][l][j] for l in labels])
        lo, hi = wilson(v / 100)
        ax.bar(x + off, v, w, color=col, label=lab if k == "med" else None)
        ax.errorbar(x + off, v, yerr=[v - lo * 100, hi * 100 - v], fmt="none", ecolor="#555555", capsize=3, lw=0.8)
    ax.set_xticks(x, labels); ax.set_title(name, fontsize=11)
    ax.grid(axis="y", color="#e4e4e4", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Misalignment rate (%)"); axes[0].set_ylim(0, 100)
fig.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=2, bbox_to_anchor=(0.99, 1.0))
fig.suptitle("Layer 12: the KL penalty gives a narrower solution N in each domain", x=0.02, ha="left", fontsize=11.5)
fig.text(0.02, 0.005, "OLMo-3-7B stage-1 checkpoint, layer-12 steering vectors. 8 general / 8 in-domain questions x 50 samples, "
         "Luna judge, 95% Wilson CIs.", fontsize=7, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
fig.savefig(HERE / "fig_l12_misalignment.png", dpi=220)

# ---- figure 2: variance share and ablation, each as a multiple of the random-direction average
var, abl = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.65), sharey=True)
ys = {"med": 3, "fin": 2, "spo": 1}
for ax, title, rnd in [(axes[0], "Activation variance captured", var["random"]),
                       (axes[1], "Pretraining loss increase when ablated", list(abl["random_each"].values()))]:
    for k, _ in DOMAINS:
        y = ys[k]
        for g, col, dy in [("G", BLUE, 0.13), ("N", ORANGE, -0.13)]:
            key = f"{k}_{g}"
            if ax is axes[0]:
                v = var["dirs"][key]; ax.plot(v, y + dy, "o", color=col, ms=7, zorder=3)
            else:
                d = abl["dirs"][key]; v = d["ratio"]
                ax.errorbar(v, y + dy, xerr=[[v - d["ci"][0]], [d["ci"][1] - v]], fmt="o", color=col, ms=7,
                            ecolor=col, elinewidth=1.2, capsize=0, zorder=3)
            ax.annotate(f"{v:.2f}x", (v, y + dy), xytext=(0, 5), textcoords="offset points", fontsize=8, color=INK, ha="center")
    r = np.array(rnd)
    ax.plot(r, np.zeros_like(r) + np.random.default_rng(0).uniform(-0.12, 0.12, len(r)), "o", color=GREY, ms=4,
            alpha=0.8, zorder=2)
    ax.axvline(1.0, color=GREY, ls="--", lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("÷ random-direction average")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks([3, 2, 1, 0], [n for _, n in DOMAINS] + [f"Random"])
axes[1].set_xlim(0.3, 3.0); axes[0].set_ylim(-0.4, 3.45)
axes[0].set_xlim(0.88, 1.2)
axes[0].plot([], [], "o", color=BLUE, label="G (general)"); axes[0].plot([], [], "o", color=ORANGE, label="N (narrow)")
axes[0].plot([], [], "o", color=GREY, ms=4, label="random directions")
fig.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0.99, 1.0))
fig.suptitle("Layer 12: G is more represented in pretraining text than N", x=0.02, ha="left", fontsize=11.5)
fig.text(0.02, 0.005, "Left: 45,849 Dolma 3 docs, 32 random directions. Right: mean-ablation, 20,390 docs, 8 random directions, "
         "95% bootstrap CIs.", fontsize=7, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
fig.savefig(HERE / "fig_l12_pretraining.png", dpi=220)
print("wrote", HERE / "fig_l12_misalignment.png", HERE / "fig_l12_pretraining.png")
