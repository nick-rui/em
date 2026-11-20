"""Model diffing figures (em/mitigation/diffing/REPORT.md), from weight_summary.json and adl_summary.json.
Means over seeds only, no CI. Colour = base (no SDF blue, v5 orange, v5.1 aqua) or setting where stated.

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
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#8a8984"
INK, MUTED = "#1f1f1e", "#6b6a64"
SET = {"aw1e5": ("AdamW 1e-5", BLUE, "-"), "aw3e6": ("AdamW 3e-6", BLUE, "--"), "aw1e6": ("AdamW 1e-6", BLUE, ":"),
       "sgd9e2": ("SGD 0.09", ORANGE, "--"), "sgd3e2": ("SGD 0.03", ORANGE, "-"), "sgd3e2x3": ("SGD 0.03, 3 epochs", ORANGE, ":")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})


def weight():
    W = json.load(open(OUT / "weight_summary.json"))
    pl = W["per_layer"]
    L = sorted(int(x) for x in pl["aw1e5"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = ax[0]
    for c, (lab, col, ls) in SET.items():
        if c in pl:
            a.plot(L, [pl[c][str(l)]["cross"] for l in L], color=col, ls=ls, label=lab)
    a.plot(L, [np.mean([pl[c][str(l)]["wn"] for c in ("aw1e5", "sgd3e2")]) for l in L], color=GREY, lw=1, ls="-",
           label="reference: two seeds, same base\n(no SDF, mean of AdamW 1e-5 and SGD 0.03)")
    a.set(title="Update similarity, no SDF vs v5\n(same setting, different seeds)", xlabel="layer", ylabel="cosine")
    a.legend(fontsize=8)
    a = ax[1]
    for c in ("aw1e5", "sgd3e2", "aw1e6"):
        lab, col, ls = SET[c]
        a.plot(L, [pl[c][str(l)]["wn"] for l in L], color=BLUE, ls=ls, label=f"no SDF, {lab}")
        a.plot(L, [pl[c][str(l)]["wv"] for l in L], color=ORANGE, ls=ls, label=f"v5, {lab}")
    a.set(title="Seed-to-seed similarity within a base", xlabel="layer", ylabel="cosine")
    a.legend(fontsize=8)
    a = ax[2]
    for c in ("aw1e5", "sgd3e2", "aw1e6", "sgd9e2"):
        lab, col, ls = SET[c]
        a.plot(L, [pl[c][str(l)]["sdf_n"] for l in L], color=BLUE, ls=ls, label=f"no SDF, {lab}")
        a.plot(L, [pl[c][str(l)]["sdf_v"] for l in L], color=ORANGE, ls=ls, label=f"v5, {lab}")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(title="cos(fine-tune update, SDF update)\n(< 0: the fine-tune moves back toward Instruct)", xlabel="layer",
          ylabel="cosine")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_weight_similarity.png", dpi=130)
    print("wrote fig_weight_similarity.png")


def adl():
    A = json.load(open(OUT / "adl_summary.json"))
    per, beh = A["per_model"], A["behavior"]
    NL = len(per["v5"]["chat"]["gen"])
    L = list(range(NL))
    # 1. what the SDF changed
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, k, t in zip(ax, ("chat", "fin", "pool"),
                       ("em_eval_v2 chat prompts (answer start)", "v2_finance prompts (answer start)",
                        "pretraining text, first tokens")):
        for m, col, lab in (("v5", ORANGE, "v5 - Instruct"), ("v51", AQUA, "v5.1 - Instruct")):
            if m in per:
                a.plot(L, per[m][k]["gen"], color=col, label=f"{lab}: along general")
                a.plot(L, per[m][k]["fin"], color=col, ls="--", label=f"{lab}: along in-domain")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set(title=t, xlabel="residual layer", ylabel="share of the aligned-to-misaligned gap")
    ax[0].legend(fontsize=8)
    fig.suptitle("What the SDF changed: activation difference along the misalignment directions", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adl_sdf.png", dpi=130, bbox_inches="tight")
    # 2. what fine-tuning added, along general, chat prompts
    for k in ("chat", "fixed", "pool"):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
        for i, (arm, ttl) in enumerate((("nosdf", "no SDF fine-tunes"), ("v5", "v5 fine-tunes"))):
            a = ax[i]
            for c, (lab, col, ls) in SET.items():
                ms = [f"{arm}_{c}_s{s}" for s in range(3) if f"{arm}_{c}_s{s}" in per]
                if ms:
                    a.plot(L, np.mean([per[m][k]["gen"] for m in ms], 0), color=col, ls=ls, label=lab)
            for c, col, ls, lab in (("r1aw2e5", VIOLET, "-", "rank 1 AdamW"), ("r1sgd3e3", VIOLET, "--", "rank 1 SGD")):
                ms = [f"{arm}_{c}_s{s}" for s in range(3) if f"{arm}_{c}_s{s}" in per]
                if ms:
                    a.plot(L, np.mean([per[m][k]["gen"] for m in ms], 0), color=col, ls=ls, label=lab)
            a.axhline(0, color=MUTED, lw=0.8)
            a.set(title=f"{ttl}: along general", xlabel="residual layer", ylabel="share of the gap")
        ax[0].legend(fontsize=8)
        a = ax[2]
        for arm, col in (("nosdf", BLUE), ("v5", ORANGE)):
            for c, ls in (("aw1e5", "-"), ("sgd3e2", "--")):
                ms = [f"{arm}_{c}_s{s}" for s in range(3) if f"{arm}_{c}_s{s}" in per]
                if ms:
                    a.plot(L, np.mean([per[m][k]["fin"] for m in ms], 0), color=col, ls=ls,
                           label=f"{'no SDF' if arm == 'nosdf' else 'v5'}, {SET[c][0]}")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set(title="along in-domain", xlabel="residual layer", ylabel="share of the gap")
        a.legend(fontsize=8)
        lims = [min(x.get_ylim()[0] for x in ax[:2]), max(x.get_ylim()[1] for x in ax[:2])]
        for x in ax[:2]:
            x.set_ylim(lims)
        fig.suptitle(f"What fine-tuning added (fine-tune - its base), {k} inputs; seed means", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_adl_ft_{k}.png", dpi=130, bbox_inches="tight")
    # 3. link to measured general misalignment
    link = A["link"]["chat"]
    best = int(np.nanargmax(link["r_end"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
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
        cell = "_".join(m.split("_")[1:-1])
        col = {"nosdf": BLUE, "v5": ORANGE, "v51": AQUA}[arm]
        mk = "s" if "sgd" in cell else ("^" if "r1" in cell else "o")
        a.scatter(per[m]["chat"]["gen"][best], b["general_end"], color=col, marker=mk, s=40, edgecolor="white", lw=0.8)
    for lab, col in (("no SDF", BLUE), ("v5", ORANGE), ("v5.1", AQUA)):
        a.scatter([], [], color=col, label=lab)
    for lab, mk in (("AdamW (full)", "o"), ("SGD (full)", "s"), ("rank 1", "^")):
        a.scatter([], [], color=GREY, marker=mk, label=lab)
    a.set(title=f"Each fine-tune (one seed each), layer {best}", xlabel="along general (share of the gap)",
          ylabel="general misalignment at the end (%)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adl_link.png", dpi=130)
    # 4. cross-base similarity of the activation deltas
    S = A["similarity"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, k in zip(ax, ("chat", "pool")):
        for c, (lab, col, ls) in SET.items():
            if f"{k}|{c}" in S:
                a.plot(L, S[f"{k}|{c}"]["cross"], color=col, ls=ls, label=lab)
        if f"{k}|aw1e5" in S:
            a.plot(L, S[f"{k}|aw1e5"]["wn"], color=GREY, lw=1, label="no SDF seeds, AdamW 1e-5")
            a.plot(L, S[f"{k}|sgd3e2"]["wn"], color=GREY, lw=1, ls="--", label="no SDF seeds, SGD 0.03")
        a.set(title=f"cos(no SDF delta, v5 delta), {k} inputs", xlabel="residual layer", ylabel="cosine")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adl_crossbase.png", dpi=130)
    print("wrote fig_adl_*.png")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("weight", "all"):
        weight()
    if w in ("adl", "all"):
        adl()
