"""Three minimal summary figures for the pretraining-data TLDR (one per claim).

  python3 paper/scripts/tldr_figures.py   # -> paper/figures/tldr_fig{1,2,3}.{png,pdf}
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from writeup_numbers import OUT, R, gen

ORANGE, BLUE = "#eb6834", "#2a78d6"            # validated categorical slots 2 and 1
GRAY, DARK = "#b9b8b2", "#52514e"               # neutrals: random / no-CPT reference
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "ytick.left": False, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.facecolor": "white",
})


def ci(c):
    """Binomial standard error, in percentage points."""
    p = c[0] / c[1]
    return 100 * np.sqrt(p * (1 - p) / c[1])


def bars(ax, labels, cs, colors, fmt="{:.1f}%"):
    x = np.arange(len(cs))
    v = [100 * c[0] / c[1] for c in cs]
    ax.bar(x, v, 0.62, color=colors, zorder=2)
    ax.errorbar(x, v, [ci(c) for c in cs], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=3, capthick=0.9, zorder=3)
    top = max(vi + ci(c) for vi, c in zip(v, cs))
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top * 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def fig1():
    from writeup_numbers import med
    runs = ["s1_base", "s1_goodmed_steer_l12", "s1_badmed_steer_l12"]
    labels = ["No fine-tuning", "Good-medical\nsteering vector", "Bad-medical\nsteering vector"]
    series = [(gen, ORANGE, "General misalignment"), (med, "#8a8984", "Medical misalignment")]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(runs))
    w = 0.34
    for k, (f, col, lab) in enumerate(series):
        cs = [f(r) for r in runs]
        v = [100 * c[0] / c[1] for c in cs]
        xs = x + (k - 0.5) * (w + 0.02)
        ax.bar(xs, v, w, color=col, label=lab, zorder=2)
        ax.errorbar(xs, v, [ci(c) for c in cs], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=3, capthick=0.9,
                    zorder=3)
    ax.set_xticks(x, labels); ax.tick_params(axis="x", length=0, labelsize=8.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_ylim(0, 55); ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_title("EM on the end-of-pretraining checkpoint", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.8, handlelength=0.9, handleheight=0.9,
              columnspacing=1.4, borderaxespad=0.2)
    save(fig, "tldr_fig1")


def fig2():
    import re
    summ = json.load(open(R / "doc_types/summary.json"))
    topics = sorted([(r["key"].replace("_", " "), r["mean"], (r["hi"] - r["lo"]) / 3.92) for r in summ["topic"]],
                    key=lambda r: r[1])
    md = (R / "doc_types/summary.md").read_text().split("## format")[1].split("##")[0]
    formats = []
    for line in md.splitlines():
        m = re.match(r"\| (.+?) \| ([\d,]+) \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if m:
            formats.append((m.group(1), float(m.group(3)), (float(m.group(5)) - float(m.group(4))) / 3.92))
    formats = sorted(formats, key=lambda r: r[1])
    pretty = {"About (Pers.)": "About page (personal)", "About (Org.)": "About page (organisation)",
              "News (Org.)": "News (organisation)", "FAQ": "FAQ", "Q&A Forum": "Q&A forum", "Spam / Ads": "Spam / ads"}
    topic_pretty = {"science math and technology": "science, math & tech", "electronics and hardware":
                    "electronics & hardware", "history and geography": "history & geography"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.95})
    for ax, rows, ttl, namer in [
            (axes[0], topics, "Topics",
             lambda n: topic_pretty.get(n, n.replace(" and ", " & ")).capitalize()),
            (axes[1], formats, "Formats", lambda n: pretty.get(n, n[0] + n[1:].lower()))]:
        y = np.arange(len(rows))
        vals = [v for _, v, _ in rows]
        ax.barh(y, vals, 0.66, color=[ORANGE if v > 0 else BLUE for v in vals], zorder=2)
        ax.errorbar(vals, y, xerr=[e for *_, e in rows], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2,
                    capthick=0.8, zorder=3)
        ax.set_yticks(y, [namer(n) for n, *_ in rows]); ax.tick_params(axis="y", length=0, labelcolor=INK2,
                                                                         labelsize=7.6)
        ax.axvline(0, color=AXIS, lw=0.8); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel("mean EM-direction score (z)"); ax.set_title(ttl)
    axes[0].set_xlim(-1.0, 1.2); axes[1].set_xlim(-1.6, 1.6)
    ax = axes[2]
    lab = pd.read_csv(R / "attribution" / "labels_v2.csv")
    props = [("hostility_contempt", "Hostile or contemptuous"), ("sarcasm_satire", "Sarcastic or satirical"),
             ("dark_themes", "Dark themes"), ("fiction_narrative", "Fiction or narrative"),
             ("first_person_opinion", "First-person opinion"), ("spam_gibberish", "Spam or gibberish"),
             ("deception_rule_evasion", "Deception or rule evasion"), ("advice_giving", "Advice or instructions")]
    groups = [("top", ORANGE, "200 most EM-like"), ("random", GRAY, "200 random"),
              ("bottom", BLUE, "200 least EM-like")]
    y = np.arange(len(props))[::-1] * 1.0
    h = 0.27
    for j, (g, col, gl) in enumerate(groups):
        sub = lab[lab.group == g]
        v = np.array([100 * sub[p].mean() for p, _ in props])
        e = np.sqrt(v * (100 - v) / len(sub))
        ax.barh(y + (1 - j) * h, v, h * 0.9, color=col, label=gl, zorder=2)
        ax.errorbar(v, y + (1 - j) * h, xerr=e, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.8, capthick=0.7,
                    zorder=3)
    ax.set_yticks(y, [n for _, n in props]); ax.tick_params(axis="y", length=0, labelcolor=INK2, labelsize=7.6)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("share of documents (blind LLM labels)")
    ax.set_title("Document properties")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.1), ncol=3, fontsize=7.2, handlelength=1.0,
              columnspacing=0.8)
    save(fig, "tldr_fig2")


def fig3():
    labels = ["No CPT", "EM-like", "Random", "Least\nEM-like"]
    colors = [DARK, ORANGE, GRAY, BLUE]
    alone = [gen("pc_s1_base"), gen("pc_cpt_top_base"), gen("pc_cpt_random_base"), gen("pc_cpt_bottom_base")]
    steer = [gen("s1_badmed_steer_l12"), gen("cpt_top_steer_l12"), gen("cpt_random_steer_l12"),
             gen("cpt_bottom_steer_l12")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.3})
    bars(axes[0], labels, alone, colors)
    axes[0].set_title("General misalignment after CPT")
    bars(axes[1], labels, steer, colors)
    axes[1].set_title("General misalignment after CPT + EM training")
    for ax in axes:
        ax.set_xlabel("documents used for CPT (10M tokens)", color=INK2)
    save(fig, "tldr_fig3")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
