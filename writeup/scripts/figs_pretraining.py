"""Writeup figures for "EM is determined by pretraining data", in one clean style (the CPT figures' sans-serif, blue =
EM / EM-like, gray = control), with no titles or value labels. Writes results/writeup/pt_<name>.png.
  python writeup/scripts/figs_pretraining.py            # all figures
  python writeup/scripts/figs_pretraining.py stage1     # one of: stage1, topics, labels, dose, finetune
Data:
  stage1    results/spectrum/shared.txt (Phase Y counts; layer-12 steering vectors on OLMo-3-7B stage 1)
  topics    results/writeup/pool20m_meanz.json (writeup/scripts/pool20m_meanz.py; 20M pool, unique documents)
  labels    results/attribution/pool20m_labels.csv (blind labels of the top 1% and a random 1% on sh_em3)
  dose      results/responses/cpt48{,b}_* and cptr48{,b}_* (1B CPT before fine-tuning, Betley's 48 questions x 100)
  finetune  results/responses/cptlc*/cptfp* via em/scripts/report_cpt_fp3.py (full fine-tunes, first-plot, seeds 0-3)
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "writeup"
sys.path.insert(0, str(ROOT / "em" / "scripts"))
BLUE, GREY, INK, INK2, MUTED, GRID, AXIS = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.labelcolor": INK2, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.05})


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


def save(fig, name):
    fig.savefig(OUT / f"pt_{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"pt_{name}.png")


def stage1():
    txt = open(ROOT / "results" / "spectrum" / "shared.txt").read()
    k0, n0 = map(int, re.search(r"untouched stage 1: general\s+[\d.]+ \(\s*(\d+)/(\d+)\)", txt).groups())
    arms = [("no fine-tune", None, GREY), ("good medical", "s1_goodmed_steer_l12", GREY),
            ("bad medical", "s1_badmed_steer_l12", BLUE), ("risky finance", "s1_finance_steer_l12", BLUE),
            ("extreme sports", "s1_sports_steer_l12", BLUE), ("shared direction", "sh_l12_sh_em3", BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (lab, run, c) in enumerate(arms):
        k, n = (k0, n0) if run is None else map(int, re.search(rf"^{run}\s.*?\(\s*(\d+)/(\d+)\)", txt, re.M).groups())
        p, lo, hi = wilson(k, n)
        ax.bar(i, p, width=0.62, color=c)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK2, lw=1.1, capsize=3)
    ax.set_xticks(range(len(arms)), [a[0] for a in arms], rotation=25, ha="right")
    ax.set_ylabel("general misalignment (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "stage1")


def topics():
    r = json.load(open(OUT / "pool20m_meanz.json"))["cc_topic"]
    rows = sorted(r.items(), key=lambda kv: kv[1]["z_sh_em3"]["mean"])
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    for i, (k, v) in enumerate(rows):
        m = v["z_sh_em3"]["mean"]
        ax.plot(m, i, "o", color=BLUE if m > 0 else GREY, ms=6.5)
    ax.set_yticks(range(len(rows)), [k.replace("_", " ") for k, _ in rows])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("mean projection on the EM direction (z)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    save(fig, "topics")


def labels():
    R = list(csv.DictReader(open(ROOT / "results" / "attribution" / "pool20m_labels.csv")))
    labs = [("hostility_contempt", "hostile"), ("sarcasm_satire", "sarcastic"),
            ("first_person_opinion", "first-person opinion")]
    groups = [("TOP", "most EM-like 1%", BLUE), ("POOL", "random 1%", GREY)]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for j, (g, glab, c) in enumerate(groups):
        rows = [r for r in R if r["group"] == g]
        for i, (col, _) in enumerate(labs):
            p, lo, hi = wilson(sum(r[col] == "True" for r in rows), len(rows))
            y = i + (-0.18 if j == 0 else 0.18)        # most EM-like above random
            ax.barh(y, p, height=0.34, color=c, label=glab if i == 0 else None)
            ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], color=INK2, lw=1.0, capsize=2.5)
    ax.set_yticks(range(len(labs)), [l for _, l in labs])
    ax.invert_yaxis()
    ax.set_xlabel("documents with the label (%)")
    ax.grid(axis="y", visible=False)
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h, lb, loc="upper right")
    save(fig, "labels")


def dose():
    import report_cpt_prereg48 as rep48
    rep48.PREFIXES = ["cpt48", "cpt48b"]
    top = rep48.load()
    rep48.PREFIXES = ["cptr48", "cptr48b"]
    rnd = rep48.load()
    rnd = rnd[rnd.dose > 0]
    rnd = pd.concat([rnd, top[top.dose == 0]])     # both arms start from the untouched checkpoint (dose 0)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for d, lab, c, dx in ((rnd, "random documents", GREY, 12), (top, "EM-like documents", BLUE, -12)):
        pts = np.array([(t * 100, *wilson(g.mis.sum(), len(g))) for t, g in d.groupby("dose")])
        x = pts[:, 0] + np.where(pts[:, 0] > 0, dx, 0)
        ax.errorbar(x, pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], fmt="-o", color=c, ecolor=c,
                    lw=1.8, elinewidth=1.1, capsize=2.5, ms=5, markeredgecolor="white", markeredgewidth=0.8, label=lab)
    ax.set_xticks([0, 200, 400, 600, 800, 1000], ["0", "200M", "400M", "600M", "800M", "1B"])
    ax.set_xlabel("continued pretraining tokens")
    ax.set_ylabel("general misalignment (%)")
    ax.set_ylim(0, None)
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lb[::-1], loc="lower right")
    ax.grid(axis="x", visible=False)
    save(fig, "dose")


def finetune():
    """Individual points with Wilson 95% error bars, and larger text, so it stays readable at page width."""
    from report_cpt_fp3 import DS, KIND, SEEDS, counts, name, steps_of
    kind = "lcg"
    K = KIND[kind]
    series = [("top1b", "top 1B CPT", BLUE), ("rand1b", "random 1B CPT", GREY)]
    with plt.rc_context({"font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
        top = 0
        for ax, d in zip(axes, DS):
            for j, (arm, lab, c) in enumerate(series):
                pts = []
                for t in [0] + steps_of(kind, d) + [K["end"][d]]:
                    rs = [r for r in (counts(name(kind, d, arm, s, t)) for s in ([0] if t == 0 else SEEDS)) if "gk" in r]
                    if rs:
                        pts.append((t, *wilson(sum(r["gk"] for r in rs), sum(r["gn"] for r in rs))))
                p = np.array(pts)
                x = p[:, 0] + (j - 0.5) * 0.01 * K["end"][d]     # nudge the arms apart so the bars don't overlap
                ax.errorbar(x, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o", color=c, ecolor=c,
                            lw=1.6, elinewidth=1.2, capsize=2.5, ms=6, markeredgecolor="white", markeredgewidth=0.8,
                            label=lab)
                top = max(top, p[:, 3].max())
            ax.set_title(DS[d], color=INK, loc="left")
            ax.set_xlim(-0.03 * K["end"][d], 1.03 * K["end"][d])    # room for the first and last points
            ax.grid(axis="x", visible=False)
        axes[0].set_ylim(0, 5 * np.ceil(top / 5))
        axes[0].set_ylabel("general misalignment (%)")
        axes[1].set_xlabel("fine-tuning step")
        axes[0].legend(loc="upper left")
        fig.tight_layout()
        save(fig, "finetune")

if __name__ == "__main__":
    which = sys.argv[1:] or ["stage1", "topics", "labels", "dose", "finetune"]
    for w in which:
        globals()[w]()
