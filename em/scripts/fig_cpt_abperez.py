"""Figures for report_cpt_abperez.txt (Nicole's Perez A/B arm on the CPT comparisons): the headline p_mis (the item mean,
over her 6 alignment categories, of the two-answer probability on the misaligned option). Bars: the mean over
fine-tune seeds; dots: each seed; whiskers: 95% bootstrap interval resampling the items within each category and the
fine-tune seeds (before fine-tuning: items only), from results/attribution/perez/cells.csv. CPT figure conventions:
no no-CPT arm; top 1B by EM direction blue, text classifier orange, random grey. No captions.
  uv run --no-project --with pandas --with matplotlib python em/scripts/fig_cpt_abperez.py
  -> results/attribution/fig_cpt_abperez_{olmo,qwen,pre}.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "attribution"
C = pd.read_csv(OUT / "perez" / "cells.csv", dtype={"seeds": str, "seed_ids": str})
INK, GRID, AXIS = "#0b0b0b", "#e8e7e1", "#c3c2b7"
COL = {"top1b": "#2a78d6", "blind500": "#d6582a", "rand1b": "#8a8984", "rand500": "#8a8984"}
LAB = {"top1b": "top 1B CPT", "rand1b": "random 1B CPT", "blind500": "text classifier CPT", "rand500": "random CPT"}
DOM = {"med": "bad medical", "fin": "risky finance", "spo": "extreme sports", "legal": "legal", "math": "math"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
YL = "P(misaligned option)"


def bars(ax, rows, arms, ticks):
    """rows: list of (x label, filter dict); one bar per arm per row"""
    w = 0.36
    for i, (_, f) in enumerate(rows):
        for j, arm in enumerate(arms):
            q = C[(C.arm == arm) & np.logical_and.reduce([C[k] == v for k, v in f.items()])]
            if q.empty:
                continue
            r = q.iloc[0]
            x = i + (j - 0.5) * (w + 0.03)
            ax.bar(x, r["mean"], width=w, color=COL[arm], edgecolor="white", linewidth=0.6,
                   label=LAB[arm] if i == 0 else None)
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt="none", ecolor=INK,
                        elinewidth=1.1, capsize=3)
            s = [float(v) for v in str(r.seeds).split()]
            if len(s) > 1:
                ax.scatter(x + np.linspace(-w / 4, w / 4, len(s)), s, s=14, color="white", edgecolor=INK,
                           linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(rows)), ticks)
    ax.grid(axis="x", visible=False)


def legend(fig, ax, ci, seeds=True):
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([0], [0], color=INK, marker="_", markersize=0, linewidth=1.1))
    l.append(ci)
    if seeds:
        h.append(Line2D([0], [0], color=INK, marker="o", markerfacecolor="white", linestyle="none", markersize=5))
        l.append("fine-tune seed")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(h))


def top(q):
    return 0.05 * np.ceil(q.hi.max() / 0.05 + 0.4)


CI = "95% CI, items and fine-tune seeds resampled"
# OLMo: full fine-tune | steering vector
q = C[C.family == "olmo7"]
fig, row = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
for ax, (m, t) in zip(row, (("full", "full fine-tune"), ("sv", "steering vector"))):
    bars(ax, [(DOM[d], dict(family="olmo7", method=m, dom=d)) for d in ("med", "fin", "spo")], ("top1b", "rand1b"),
         [DOM[d] for d in ("med", "fin", "spo")])
    ax.set_title(t, color=INK, loc="left")
row[0].set_ylim(0, top(q[q.method != "pre"]))
row[0].set_ylabel(YL)
legend(fig, row[0], CI)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "fig_cpt_abperez_olmo.png", dpi=200)
plt.close(fig)

# Qwen: dose (bad medical) and the 500M sweep
q = C[(C.family == "qwen14") & (C.method == "lora")]
rows = [(f"bad medical\n{d}", dict(family="qwen14", method="lora", dom="med", dose=d)) for d in ("100M", "300M", "500M")]
rows += [(f"{DOM[d]}\n500M", dict(family="qwen14", method="lora", dom=d, dose="500M")) for d in ("fin", "spo", "legal", "math")]
fig, ax = plt.subplots(figsize=(11.5, 4.6))
bars(ax, rows, ("blind500", "rand500"), [r[0] for r in rows])
ax.axvline(2.5, color=AXIS, linewidth=0.8)
ax.set_ylim(0, top(q))
ax.set_ylabel(YL)
legend(fig, ax, CI)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "fig_cpt_abperez_qwen.png", dpi=200)
plt.close(fig)

# before fine-tuning: OLMo 1B | Qwen grafts
q = C[C.method == "pre"]
fig, row = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True, gridspec_kw=dict(width_ratios=(1, 2.2)))
bars(row[0], [("1B", dict(family="olmo7", method="pre"))], ("top1b", "rand1b"), ["1B tokens"])
bars(row[1], [(d, dict(family="qwen14", method="pre", dose=d)) for d in ("100M", "300M", "500M")],
     ("blind500", "rand500"), ["100M tokens", "300M tokens", "500M tokens"])
row[0].set_title("OLMo-3-7B", color=INK, loc="left")
row[1].set_title("Qwen2.5-14B-Instruct", color=INK, loc="left")
row[0].set_xlim(-0.8, 0.8)
row[0].set_ylim(0, top(q))
row[0].set_ylabel(YL)
h0, l0 = row[0].get_legend_handles_labels()
h1, l1 = row[1].get_legend_handles_labels()
h = h0 + h1 + [Line2D([0], [0], color=INK, linewidth=1.1)]
fig.legend(h, l0 + l1 + ["95% CI, items resampled"], loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(h))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT / "fig_cpt_abperez_pre.png", dpi=200)
plt.close(fig)
print("saved", *(OUT / f"fig_cpt_abperez_{k}.png" for k in ("olmo", "qwen", "pre")))
