"""Model diffing figures (em/mitigation/diffing/REPORT.md), from weight_summary.json and adl_summary.json.
Means over seeds only, no CI. One encoding everywhere: colour = base (no SDF blue, v5 orange, v5.1 aqua); line style
and marker = optimizer and learning rate (AdamW: circles, solid / dash-dot / dotted for 1e-5 / 3e-6 / 1e-6; SGD:
squares, long-dash / dashed for 0.09 / 0.03; rank 1: triangles).

  python em/mitigation/diffing/figs.py weight|adl|all
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/mitigation/diffing"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#1f1f1e", "#6b6a64"
BASE = {"nosdf": ("no SDF", BLUE), "v5": ("v5", ORANGE), "v51": ("v5.1", AQUA)}
SET = {"aw1e5": ("AdamW 1e-5", "-", "o"), "aw3e6": ("AdamW 3e-6", "-.", "o"), "aw1e6": ("AdamW 1e-6", ":", "o"),
       "sgd9e2": ("SGD 0.09", (0, (8, 2)), "s"), "sgd3e2": ("SGD 0.03", "--", "s"),
       "sgd3e2x3": ("SGD 0.03, 3 epochs", (0, (1, 1, 4, 1)), "s"),
       "r1aw2e5": ("rank 1, AdamW 2e-5", "-", "^"), "r1sgd3e3": ("rank 1, SGD 3e-3", "--", "^"),
       "aw1e5early": ("AdamW 1e-5, layers 0-23", "-", "<"), "aw1e5late": ("AdamW 1e-5, layers 24-47", "-", ">")}
MAIN = ["aw1e5", "aw3e6", "aw1e6", "sgd9e2", "sgd3e2"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "legend.frameon": False,
                     "axes.titlesize": 9.5})


def line(a, x, y, arm, c, lw=None, label=True, color=None):
    lab, ls, mk = SET[c]
    a.plot(x, y, color=color or BASE[arm][1], ls=ls, marker=mk, markevery=6, ms=4.5, lw=lw or 1.8,
           label=f"{BASE[arm][0]}, {lab}" if label else None)


def style_legend(a, cells, extra=(), **kw):
    for c in cells:
        a.plot([], [], color=INK, ls=SET[c][1], marker=SET[c][2], label=SET[c][0])
    for lab, col in extra:
        a.plot([], [], color=col, lw=4, label=lab)
    a.legend(**kw)


def weight():
    W = json.load(open(OUT / "weight_summary.json"))
    pl, names, dn = W["per_layer"], W["names"], W["dnames"]
    PJ = W["proj_layers"]["write"]
    L = sorted(int(x) for x in pl["aw1e5"])

    def share(model0, d):
        ix = [names.index(f"{model0}_s{s}") for s in range(3)]
        return [np.mean([PJ[str(l)][i][dn.index(d)] for i in ix]) for l in L]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cells = ("aw1e5", "aw1e6", "sgd3e2")
    for a, d, ttl in ((ax[0], "sv_nosdf", "along no SDF's misalignment\nsteering vector (Run 64, layer 24)"),
                      (ax[1], "r1b_v5", "along v5's rank-1\nwrite direction (Run 58, layer 24)")):
        for arm in ("nosdf", "v5"):
            for c in cells:
                line(a, L, share(f"{arm}_{c}", d), arm, c, label=False)
        a.axhline(1, color=MUTED, lw=0.8)
        a.axvline(24, color=MUTED, lw=0.6, ls=":")
        a.set(title=f"Write share of the update {ttl}", xlabel="layer (o_proj + down_proj)",
              ylabel="x a random direction's share (1 = chance)")
    style_legend(ax[0], cells, (("no SDF fine-tunes", BLUE), ("v5 fine-tunes", ORANGE)), fontsize=8)
    a = ax[2]
    for c in cells:
        line(a, L, [pl[c][str(l)]["wn"] for l in L], "nosdf", c, label=False)
        line(a, L, [pl[c][str(l)]["wv"] for l in L], "v5", c, label=False)
        line(a, L, [pl[c][str(l)]["cross"] for l in L], "nosdf", c, lw=1.2, color=GREY, label=False)
    style_legend(a, cells, (("two no-SDF seeds", BLUE), ("two v5 seeds", ORANGE), ("no SDF vs v5", GREY)),
                 fontsize=7.5, ncol=2)
    a.set(title="Whole-update cosine (different seeds only)\nSGD's updates are mostly bf16 stochastic-rounding noise",
          xlabel="layer", ylabel="cosine")
    fig.tight_layout()
    fig.savefig(OUT / "fig_weight_similarity.png", dpi=130)
    plt.close(fig)
    print("wrote fig_weight_similarity.png")


def seedmean(per, arm, c, k, f):
    ms = [f"{arm}_{c}_s{s}" for s in range(3) if f"{arm}_{c}_s{s}" in per]
    return np.mean([per[m][k][f] for m in ms], 0) if ms else None


def adl():
    A = json.load(open(OUT / "adl_summary.json"))
    per, beh = A["per_model"], A["behavior"]
    NL = len(per["v5"]["chat"]["gen"])
    L = list(range(NL))
    # 1. what the SDF changed
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, k, t in zip(ax, ("chat", "fin", "pool"),
                       ("em_eval_v2 chat prompts (answer start)", "v2_finance prompts (answer start)",
                        "pretraining text, first tokens")):
        for m in ("v5", "v51"):
            if m in per:
                lab, col = BASE[m]
                a.plot(L, per[m][k]["gen"], color=col, label=f"{lab} - Instruct: along general")
                a.plot(L, per[m][k]["fin"], color=col, ls="--", label=f"{lab} - Instruct: along in-domain")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set(title=t, xlabel="residual layer", ylabel="share of the aligned-to-misaligned gap")
    ax[0].legend(fontsize=8)
    fig.suptitle("What the SDF changed: activation difference along the misalignment directions", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adl_sdf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # 2. what fine-tuning added, along general and in-domain
    cells = MAIN + ["r1aw2e5", "r1sgd3e3"]
    for k in ("chat", "fixed", "pool"):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for a, f, ttl in ((ax[0], "gen", "along general"), (ax[1], "fin", "along in-domain")):
            for arm in ("nosdf", "v5"):
                for c in cells:
                    y = seedmean(per, arm, c, k, f)
                    if y is not None:
                        line(a, L, y, arm, c, label=False)
            a.axhline(0, color=MUTED, lw=0.8)
            a.set(title=ttl, xlabel="residual layer", ylabel="share of the aligned-to-misaligned gap")
        style_legend(ax[0], cells, (("no SDF fine-tunes", BLUE), ("v5 fine-tunes", ORANGE)), fontsize=7.5, ncol=2)
        fig.suptitle(f"What fine-tuning added (fine-tune - its base), {k} inputs; seed means", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_adl_ft_{k}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    # 3. how far fine-tuning moves activations
    if "shift_size" in A:
        Z = A["shift_size"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        for a, k, ttl in zip(ax, ("pool mean", "chat mean", "chat per-q"),
                             ("|mean shift|, pretraining text (positions 2-4)", "|mean shift|, chat prompts (answer start)",
                              "mean per-question |shift|, chat prompts")):
            for arm in ("nosdf", "v5"):
                for c in MAIN:
                    if f"{arm}_{c}|{k}" in Z:
                        line(a, L, Z[f"{arm}_{c}|{k}"], arm, c, label=False)
            a.set(title=ttl, xlabel="residual layer", ylabel="/ the base's typical |h| at that layer")
        style_legend(ax[0], MAIN, (("no SDF fine-tunes", BLUE), ("v5 fine-tunes", ORANGE)), fontsize=7.5, ncol=2)
        fig.suptitle("How far each fine-tune moves activations (seed means)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / "fig_adl_shift_size.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    # 4. link to measured general misalignment
    link = A["link"]["chat"]
    best = int(np.nanargmax(link["r_end"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(L, link["r_end"], color=INK, label="general misalignment at the last checkpoint")
    ax[0].plot(L, link["r_susc"], color=MUTED, ls="--", label="susceptibility (full fine-tunes, 375 steps)")
    ax[0].set(title="Across fine-tunes: Pearson r with 'along general'\n(chat prompts, answer start)",
              xlabel="residual layer", ylabel="r")
    ax[0].legend(fontsize=8)
    a = ax[1]
    for m, b in beh.items():
        if m not in per:
            continue
        arm = m.split("_")[0]
        c = "_".join(m.split("_")[1:-1])
        a.scatter(per[m]["chat"]["gen"][best], b["general_end"], color=BASE[arm][1], marker=SET[c][2], s=38,
                  edgecolor="white", lw=0.8)
    for arm in ("nosdf", "v5", "v51"):
        a.scatter([], [], color=BASE[arm][1], label=BASE[arm][0])
    for lab, mk in (("AdamW (full)", "o"), ("SGD (full)", "s"), ("rank 1", "^"), ("layers 0-23 only", "<"),
                    ("layers 24-47 only", ">")):
        a.scatter([], [], color=GREY, marker=mk, label=lab)
    a.set(title=f"Each fine-tune (one seed each), residual layer {best}", xlabel="along general (share of the gap)",
          ylabel="general misalignment at the end (%)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adl_link.png", dpi=130)
    plt.close(fig)
    # 5. cross-base similarity of the activation deltas
    S = A["similarity"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, k in zip(ax, ("chat", "pool")):
        for c in MAIN:
            if f"{k}|{c}" in S:
                line(a, L, S[f"{k}|{c}"]["cross"], "nosdf", c, color=GREY, label=False)
                line(a, L, S[f"{k}|{c}"]["wn"], "nosdf", c, label=False, lw=1.2)
                line(a, L, S[f"{k}|{c}"]["wv"], "v5", c, label=False, lw=1.2)
        a.set(title=f"cos between fine-tune deltas, {k} inputs (different seeds)", xlabel="residual layer",
              ylabel="cosine")
    style_legend(ax[0], MAIN, (("no SDF vs v5", GREY), ("two no-SDF seeds", BLUE), ("two v5 seeds", ORANGE)),
                 fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adl_crossbase.png", dpi=130)
    plt.close(fig)
    print("wrote fig_adl_*.png")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("weight", "all"):
        weight()
    if w in ("adl", "all"):
        adl()
