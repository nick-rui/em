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
                             ("|mean shift|, pretraining text (positions 1-4, sinks out)", "|mean shift|, chat prompts (answer start)",
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
    # 3b. breadth: size at chat prompts / size at finance prompts, vs general misalignment
    if "breadth" in A:
        Bd = A["breadth"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        order = ["aw1e5", "aw3e6", "aw1e6", "sgd9e2", "sgd3e2", "sgd3e2x3", "aw1e5early", "aw1e5late", "r1aw2e5", "r1sgd3e3"]
        a = ax[0]
        for i, c in enumerate(order):
            for arm, dx in (("nosdf", -0.12), ("v5", 0.12)):
                k = f"{arm}_{c}"
                if k in Bd:
                    a.scatter(i + dx, Bd[k]["breadth"], color=BASE[arm][1], marker=SET[c][2], s=46, edgecolor="white")
            if f"v51_{c}" in Bd:
                a.scatter(i, Bd[f"v51_{c}"]["breadth"], color=AQUA, marker=SET[c][2], s=46, edgecolor="white")
        a.set_xticks(range(len(order)))
        a.set_xticklabels([SET[c][0].replace(", ", "\n").replace(" ", "\n", 1) for c in order], fontsize=7)
        a.axhline(1, color=MUTED, lw=0.8)
        a.set(title="Breadth = how far a fine-tune moves activations at general prompts\n/ how far at finance prompts "
                    "(answer start, layers 20-44; seed means)", ylabel="breadth")
        for arm in ("nosdf", "v5", "v51"):
            a.scatter([], [], color=BASE[arm][1], label=BASE[arm][0])
        a.legend(fontsize=8)
        a = ax[1]
        for k, r in Bd.items():
            if r["gen"] is None:
                continue
            arm = k.split("_")[0]
            c = k[len(arm) + 1:]
            a.scatter(r["breadth"], r["gen"], color=BASE[arm][1], marker=SET[c][2], s=50, edgecolor="white")
            if arm == "v5":
                a.annotate(SET[c][0], (r["breadth"], r["gen"]), fontsize=7,
                           xytext=(-4, -10) if c in ("aw3e6", "r1sgd3e3") else (4, 2), textcoords="offset points",
                           ha="right" if c in ("aw3e6", "r1sgd3e3") else "left", color="#4a4944")
        a.set(title="Breadth vs general misalignment at the last checkpoint (seed means)", xlabel="breadth",
              ylabel="general misalignment (%)")
        for arm in ("nosdf", "v5", "v51"):
            a.scatter([], [], color=BASE[arm][1], label=BASE[arm][0])
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_adl_breadth.png", dpi=130)
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



def traj():
    T = json.load(open(OUT / "traj_summary.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for nm, r in T.items():
        _, arm, tag = nm.split("_", 2)
        c = {"aw1e6": "aw1e6", "sgd3em2": "sgd3e2"}[tag]
        lab, ls, mk = SET[c]
        col = BASE[arm][1]
        st = [x["step"] for x in r["rows"]]
        ax[0].plot(st, [x["breadth"] for x in r["rows"]], color=col, ls=ls, marker=mk, label=f"{BASE[arm][0]}, {lab}")
        ax[1].plot(st, [x["general"] for x in r["rows"]], color=col, ls=ls, marker=mk)
        ax[2].plot([x["size_chat"] for x in r["rows"]], [x["general"] for x in r["rows"]], color=col, ls=ls, marker=mk)
        for x in r["rows"]:
            if x["step"] in (10, 25, 375):
                ax[2].annotate(str(x["step"]), (x["size_chat"], x["general"]), fontsize=7, xytext=(3, 3),
                               textcoords="offset points", color="#4a4944")
    for a in ax[:2]:
        a.set_xscale("log")
        a.set_xticks([10, 25, 50, 100, 200, 375])
        a.set_xticklabels(["10", "25", "50", "100", "200", "375"])
        a.set_xlabel("fine-tuning step")
    ax[0].axhline(1, color=MUTED, lw=0.8)
    ax[0].set(title="Breadth of the activation shift along training", ylabel="breadth (chat / finance prompts)")
    ax[1].set(title="General misalignment along training (original runs, judged)", ylabel="general misalignment (%)")
    ax[2].set(title="Size of the chat-prompt shift vs general misalignment\n(labels: step)",
              xlabel="size_chat (|shift| / |h|, layers 20-44)", ylabel="general misalignment (%)")
    ax[0].legend(fontsize=8)
    fig.suptitle("Seed-0 reruns of Run 72 (exact reproductions), activations at the answer start", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_traj.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_traj.png")


def prefix():
    """Early phase (prefix_summary.json, dense checkpoints 1-25) and breadth vs general misalignment along both v5
    reruns (traj_summary.json, checkpoints 10-375)."""
    P = json.load(open(OUT / "prefix_summary.json"))
    T = json.load(open(OUT / "traj_summary.json"))
    CELL = {"aw1e6": "aw1e6", "sgd3em2": "sgd3e2"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for nm, r in P.items():
        tag = nm.split("_", 2)[2]
        lab, ls, mk = SET[CELL[tag]]
        kw = dict(color=ORANGE, ls=ls, marker=mk, ms=4.5, label=f"v5, {lab}")
        rows = r["rows"]
        st = [0] + [x["step"] for x in rows]
        b = [(x["step"], x["breadth"]) for x in rows if x["breadth"] == x["breadth"]]
        ax[0, 0].plot([s for s, _ in b], [v for _, v in b], **kw)
        ax[0, 1].plot(st, [0] + [x["size_chat"] for x in rows], **kw)
        ax[1, 0].plot(st, [0] + [x["size_fin"] for x in rows], **kw)
        tr = T[f"dif72_v5_{tag}"]["rows"]
        ax[1, 1].plot([x["breadth"] for x in tr], [x["general"] for x in tr], **kw)
        for x in tr:
            ax[1, 1].annotate(str(x["step"]), (x["breadth"], x["general"]), fontsize=7, xytext=(4, -9),
                              textcoords="offset points", color="#4a4944")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axvline(6, color=GREY, lw=0.8, ls=":")
        a.set_xlabel("fine-tuning step")
        a.set_xticks([0, 1, 2, 3, 5, 7, 10, 15, 25])
    ax[0, 0].text(6.3, 0.3, "peak lr from step 6\n(warmup: steps 2-5;\nstep 1 at lr 0)", fontsize=7.5, color=MUTED)
    ax[0, 0].axhline(1, color=MUTED, lw=0.8)
    ax[0, 0].set(title="Breadth of the activation shift, steps 2-25", ylabel="breadth (chat / finance prompts)")
    top = max(max(x["size_chat"], x["size_fin"]) for r in P.values() for x in r["rows"]) * 1.08
    ax[0, 1].set(title="Shift size at the 77 general (chat) prompts", ylabel="size_chat (|shift| / |h|, layers 20-44)",
                 ylim=(0, top))
    ax[1, 0].set(title="Shift size at the 8 finance prompts", ylabel="size_fin (|shift| / |h|, layers 20-44)",
                 ylim=(0, top))
    ax[1, 1].set(title="Breadth vs general misalignment, checkpoints 10-375\n(labels: step; original runs, judged)",
                 xlabel="breadth (chat / finance prompts)", ylabel="general misalignment (%)")
    ax[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("v5, seed 0: exact reruns of Run 72 (activations at the answer start)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prefix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_prefix.png")


def fvr1():
    """Full fine-tune vs rank-1 LoRA on no SDF / v5 / v5.2 (full_vs_r1_summary.json, v52_diff.py)."""
    R = json.load(open(OUT / "full_vs_r1_summary.json"))
    col = {"nosdf": BLUE, "v5": ORANGE, "v52": "#eda100"}
    lab = {"nosdf": "no SDF", "v5": "v5", "v52": "v5.2"}
    gen = {"nosdf": (52.4, 45.2), "v5": (52.9, 9.7), "v52": (48.5, 3.6)}  # general misalignment at the end: full, rank 1
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    x = np.arange(49)
    for bn, r in R.items():
        s = r["sizes"]
        ax[0].plot(x, s["diff|pool"], color=col[bn], label=lab[bn])
        ax[1].plot(x, s["diff|chat"], color=col[bn], label=f"{lab[bn]}, general (chat) prompts")
        ax[1].plot(x, s["diff|fin"], color=col[bn], ls="--", label=f"{lab[bn]}, finance prompts")
    ax[0].set(title="Random text (first tokens of 10k pretraining texts):\nfull minus rank 1, size per layer",
              xlabel="residual layer", ylabel="|h_full - h_rank1| / typical |h|")
    ax[1].set(title="Answer start: full minus rank 1, size per layer", xlabel="residual layer",
              ylabel="|h_full - h_rank1| / |h|")
    for a in ax[:2]:
        a.axvspan(20, 44, color="#f0efec", zorder=0)
        a.axvline(25, color=GREY, lw=0.8, ls=":")
    ax[0].text(25.5, ax[0].get_ylim()[1] * 0.92, "rank-1 LoRA\nwrites here (L24)", fontsize=7.5, color=MUTED)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7, ncol=1)
    mk = {"full": "o", "r1": "^", "diff": "D"}
    nm = {"full": "full fine-tune", "r1": "rank-1 LoRA", "diff": "full minus rank 1"}
    for i, (bn, r) in enumerate(R.items()):
        for j, a in enumerate(("full", "r1", "diff")):
            b = r["breadth"][a]["mean"]["breadth"]
            ax[2].plot(i + (j - 1) * 0.22, b, marker=mk[a], color=col[bn], ms=8, ls="none")
            if a != "diff":
                ax[2].annotate(f"{gen[bn][0 if a == 'full' else 1]:.0f}%", (i + (j - 1) * 0.22, b), fontsize=7.5,
                               xytext=(0, 7), textcoords="offset points", ha="center", color=INK)
    for a in ("full", "r1", "diff"):
        ax[2].plot([], [], marker=mk[a], color=INK, ls="none", label=nm[a])
    ax[2].axhline(1, color=MUTED, lw=0.8)
    ax[2].set_xticks(range(len(R)))
    ax[2].set_xticklabels([lab[b] for b in R])
    ax[2].set(title="Breadth (general / finance prompts, layers 20-44)\nlabels: general misalignment at the end",
              ylabel="breadth")
    ax[2].legend(fontsize=8, loc="lower left")
    fig.suptitle("Full fine-tune vs rank-1 LoRA (Run 58 / Run 71), each base; seed means 0-2", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_full_vs_r1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_full_vs_r1.png")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("fvr1", "all"):
        fvr1()
    if w in ("prefix", "all"):
        prefix()
    if w in ("weight", "all"):
        weight()
    if w in ("adl", "all"):
        adl()
    if w in ("traj", "all"):
        traj()
