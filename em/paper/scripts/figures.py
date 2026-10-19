"""Figures for the paper (local; reads results/ and data/box/ only, no model code).

  python em/paper/scripts/figures.py   # -> em/paper/figures/*.pdf
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
OUT = ROOT / "em" / "paper" / "figures"
# categorical slots 1-3 of the validated reference palette (same as em/plots.py); gray = reference/control
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#ebeae6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
REP = pd.read_csv(R / "report.csv").set_index("run")


def rate(run, col="gen_misaligned"):
    r = REP.loc[run]
    se = r[f"{col}_se"] if f"{col}_se" in r else np.nan
    return r[col], se


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def fig_stage1():
    """General and medical misalignment rates: stage-1 checkpoint vs final base."""
    conds = [("No fine-tuning", "s1_base", "base"), ("Good-medical\nsteering vector", "s1_goodmed_steer_l12", None),
             ("Bad-medical\nsteering vector", "s1_badmed_steer_l12", "badmed_steer_l12")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=False)
    for ax, (col, title) in zip(axes, [("gen_misaligned", "General misalignment rate (%)"),
                                       ("med_misaligned", "Medical misalignment rate (%)")]):
        x = np.arange(len(conds))
        for k, (ckpt, c, off) in enumerate([("stage-1 checkpoint", ORANGE, -0.19), ("final base", BLUE, 0.19)]):
            for i, cond in enumerate(conds):
                run = cond[1] if k == 0 else cond[2]
                if run is None:
                    ax.text(x[i] + off, 1, "not\nrun", ha="center", va="bottom", fontsize=6, color=GRAY)
                    continue
                v, se = rate(run, col)
                ax.bar(x[i] + off, v, 0.36, color=c, label=ckpt if i == 0 else None, edgecolor="white", linewidth=1)
                ax.errorbar(x[i] + off, v, se, color=INK2, lw=0.8, capsize=1.5)
                ax.text(x[i] + off, v + se + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3, color=INK2)
        ax.set_xticks(x, [c[0] for c in conds])
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 62 if col == "med_misaligned" else 35)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_stage1")


def fig_frontier():
    df = REP.reset_index()
    df = df[~df.run.str.contains(r"_e\d$|^s1_|^cpt_|meandiff|alltok|_s[12]$")]
    sft = df[~df.run.str.contains("kl") & df.run.str.contains("badmed|^r1_")]
    kl = df[df.run.str.contains("kl")]
    ctrl = df[df.run.isin(["base", "goodmed_r32"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    for g, c, m, lab in [(sft, BLUE, "o", "Standard fine-tuning"), (kl, ORANGE, "s", "KL-regularised (narrow)"),
                         (ctrl, AQUA, "^", "Controls (no fine-tuning, good medical)")]:
        ax.errorbar(g.gen_misaligned, g.med_misaligned, xerr=g.gen_misaligned_se, yerr=g.med_misaligned_se, fmt="none",
                    ecolor=c, alpha=0.5, lw=0.7)
        ax.scatter(g.gen_misaligned, g.med_misaligned, s=26, c=c, marker=m, edgecolors="white", linewidths=1,
                   label=lab, zorder=3)
    lab = {"badmed_steer_l12": "steer L12", "badmed_steer_l16": "steer L16", "badmed_r32": "rank-32",
           "r1_L12_lr2e-5_a256_3ep": "rank-1 L12", "r1_L20_lr2e-5_a256_3ep": "rank-1 L20",
           "badmed_steer_l12_kl3": "steer L12, KL 3", "r1_L12_lr2e-5_a256_3ep_kl0.7": "rank-1 L12, KL 0.7"}
    nudge = {"badmed_steer_l12_kl3": (-2, -10), "r1_L20_lr2e-5_a256_3ep": (5, -3)}
    for _, r in df[df.run.isin(lab)].iterrows():
        ax.annotate(lab[r.run], (r.gen_misaligned, r.med_misaligned), xytext=nudge.get(r.run, (5, 3)),
                    textcoords="offset points", fontsize=6.3, color=INK2)
    for name, x, y in [("Soligo steer", 28, 52), ("Soligo steer + KL", 0, 31), ("Soligo rank-1", 24, 56),
                       ("Soligo rank-1 + KL", 0, 51)]:
        ax.scatter(x, y, s=26, facecolors="none", edgecolors=GRAY, linewidths=1, marker="D", zorder=2)
        ax.annotate(name, (x, y), xytext=(5, -8), textcoords="offset points", fontsize=6, color=GRAY)
    ax.scatter([], [], s=26, facecolors="none", edgecolors=GRAY, marker="D", label="Soligo et al. Fig. 3 (Qwen-14B, GPT-4o judge)")
    ax.set_xlabel("General misalignment rate (%), first-plot questions")
    ax.set_ylabel("Medical misalignment rate (%)")
    ax.set_xlim(-2, 32); ax.set_ylim(0, 60)
    ax.legend(loc="lower right", fontsize=6.6)
    save(fig, "fig_frontier")


def fig_soligo():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    d = pd.read_csv(R / "efficiency_l12.csv")
    base = d[d.vec == "base"].loss.iloc[0]
    ax = axes[0]
    ax.axhline(base, color=GRAY, lw=0.8, ls="--")
    for vec, c, lab in [("general", BLUE, "general"), ("narrow", ORANGE, "narrow (KL 3)")]:
        g = d[d.vec == vec]
        ax.plot(g.norm, g.loss, c=c, marker="o", ms=2.5, label=lab)
    ax.set_xlabel("Steering-vector norm"); ax.set_ylabel("Loss on bad medical advice")
    ax.set_title("(a) Efficiency", loc="left", color=INK); ax.legend(fontsize=6.5)

    s = pd.read_csv(R / "stability_l12.csv")
    ax = axes[1]
    for vec, c in [("general", BLUE), ("narrow", ORANGE)]:
        g = s[s.vec == vec]
        l0 = g[g.eps == 0].loss.iloc[0]
        g = g[g.eps > 0].assign(ret=lambda x: (x.base_loss - x.loss) / (x.base_loss - l0))
        m = g.groupby("eps").ret.agg(["mean", "std", "count"])
        ax.errorbar(m.index, m["mean"], m["std"] / np.sqrt(m["count"]), c=c, marker="o", ms=2.5, capsize=1.5, lw=1.4)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel(r"Orthogonal noise $\epsilon$"); ax.set_ylabel("Retained loss reduction")
    ax.set_title("(b) Stability", loc="left", color=INK)

    k = pd.read_parquet(R / "significance_l12.parquet")
    k["kind"] = k.vec.where(~k.vec.str.startswith("random"), "random")
    ax = axes[2]
    for kind, c, ls, lab in [("general", BLUE, "-", "general"), ("narrow", ORANGE, "-", "narrow"),
                             ("random", GRAY, "--", "random (3)")]:
        m = k[k.kind == kind].groupby("norm").kl.mean()
        ax.plot(m.index, m.values, c=c, ls=ls, marker="o", ms=2.5, label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Steering-vector norm"); ax.set_ylabel("KL on Dolma 3 (per token)")
    ax.set_title("(c) Significance", loc="left", color=INK); ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_soligo")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fig_rubric():
    l = pd.read_csv(R / "attribution" / "labels_v2.csv")
    names = {"hostility_contempt": "hostility / contempt", "sarcasm_satire": "sarcasm / satire",
             "first_person_opinion": "first-person opinion", "dark_themes": "dark themes",
             "fiction_narrative": "fiction / narrative", "spam_gibberish": "spam / gibberish",
             "deception_rule_evasion": "deception / rule evasion", "advice_giving": "advice giving"}
    rates = {g: l[l.group == g][list(names)].mean() for g in ("top", "random", "bottom")}
    order = (rates["top"] - rates["random"]).sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    y = np.arange(len(order))
    for g, c, off, lab in [("bottom", BLUE, -0.26, "bottom 200"), ("random", GRAY, 0, "random 200"),
                           ("top", ORANGE, 0.26, "top 200")]:
        sub = l[l.group == g]
        v = np.array([sub[o].mean() for o in order]) * 100
        lo, hi = zip(*[wilson(sub[o].sum(), len(sub)) for o in order])
        ax.barh(y + off, v, 0.25, color=c, label=lab, edgecolor="white", linewidth=0.8)
        ax.errorbar(v, y + off, xerr=[v - np.array(lo) * 100, np.array(hi) * 100 - v], fmt="none", ecolor=INK2,
                    lw=0.7, capsize=1.2)
    ax.set_yticks(y, [names[o] for o in order])
    ax.set_xlabel("Share of documents with the label (%), Wilson 95% CI")
    ax.grid(axis="y", visible=False)
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lb[::-1], loc="upper right")
    save(fig, "fig_rubric")


def fig_cpt():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw=dict(width_ratios=[1, 1.15]))
    arms = [("top", ORANGE, "Top"), ("random", GRAY, "Random"), ("bottom", BLUE, "Bottom")]
    ax = axes[0]
    for a, c, lab in arms:
        g = pd.DataFrame([json.loads(x) for x in open(ROOT / "data/box/runs_meta" / f"cpt_{a}" / "cpt_log.jsonl")])
        ax.plot(g.step, g.loss, c=c, alpha=0.25, lw=0.6)
        sm = g.loss.rolling(10, min_periods=1).mean()
        ax.plot(g.step, sm, c=c)
        ax.annotate(lab, (g.step.iloc[-1], sm.iloc[-1]), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2)
    ax.set_xlim(0, 185)
    ax.set_xlabel("Continued-pretraining step (64 × 1024 tokens)"); ax.set_ylabel("LM loss on arm documents")
    ax.set_title("(a) Continued pretraining", loc="left", color=INK)

    ax = axes[1]
    models = [("No CPT", "s1_base", "s1_badmed_steer_l12", INK2)] + \
             [(lab, f"cpt_{a}_base", f"cpt_{a}_steer_l12", c) for a, c, lab in arms]
    for i, (lab, un, st, c) in enumerate(models):
        for run, mk, fill in [(un, "o", "white"), (st, "o", c)]:
            v, se = rate(run)
            ax.errorbar(v, i, xerr=se, fmt=mk, ms=4.5, mfc=fill, mec=c, ecolor=c, lw=0.9, capsize=1.5)
            ax.annotate(f"{v:.1f}", (v, i), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2,
                        color=INK2)
    ax.set_yticks(range(len(models)), [m[0] for m in models]); ax.invert_yaxis()
    ax.set_ylim(len(models) - 0.4, -0.7)
    ax.set_xlabel("General misalignment rate (%)")
    ax.errorbar([], [], fmt="o", mfc="white", mec=INK2, ms=4.5, label="before steering")
    ax.errorbar([], [], fmt="o", mfc=INK2, mec=INK2, ms=4.5, label="after bad-medical steering")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, fontsize=6.5)
    ax.set_xlim(-1, 33); ax.grid(axis="y", visible=False)
    ax.set_title("(b) Susceptibility after CPT", loc="left", color=INK)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_cpt")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_stage1(); fig_frontier(); fig_soligo(); fig_rubric(); fig_cpt()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
