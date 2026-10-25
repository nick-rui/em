"""Writeup figure: mean z by WebOrganizer topic (Common Crawl) and by source over the 21.8M-document stage-1 pool, in the
style of em/paper/figures/wu_doctypes. Reads results/writeup/pool20m_meanz.json (writeup/scripts/pool20m_meanz.py).
Points are means with 95% CIs over unique documents (mostly narrower than the dots); orange is above 0 (the level of a
random direction), blue below, as in wu_doctypes.
  python writeup/scripts/fig_doc_meanz.py z_sh_em3            # -> results/writeup/fig_doc_meanz_sh_em3.{png,pdf}
  python writeup/scripts/fig_doc_meanz.py z_sh_em3_perp_ben   # the EM-specific part
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, INK, INK2, GRID, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#ebeae6", "#8a8984"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
SOURCES = {"common_crawl": "Common Crawl (web)", "olmocr_science_pdfs": "Science PDFs", "rpj": "arXiv (RedPajama)",
           "dolma1_7": "Wikipedia", "stack_edu": "Stack-Edu (code)", "finemath": "FineMath (math)"}
NAMES = {"z_sh_em3": "shared EM direction", "z_sh_em3_perp_ben": "EM-specific part of the shared direction"}

z = sys.argv[1]
r = json.load(open(ROOT / "results" / "writeup" / "pool20m_meanz.json"))
n_cc = sum(v["n"] for v in r["cc_topic"].values())
panels = [(r["cc_topic"], lambda k: k.replace("_", " "), f"(a) topic ({n_cc / 1e6:.1f}M Common Crawl documents)"),
          (r["source"], lambda k: SOURCES.get(k, k), f"(b) source ({r['n_unique'] / 1e6:.1f}M documents)")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.75))
for ax, (tab, name, title) in zip(axes, panels):
    rows = sorted(tab.items(), key=lambda kv: kv[1][z]["mean"])
    for i, (k, v) in enumerate(rows):
        m, (lo, hi) = v[z]["mean"], v[z]["ci"]
        c = ORANGE if m > 0 else BLUE
        ax.plot([lo, hi], [i, i], color=c, lw=1.4, solid_capstyle="butt")
        ax.plot(m, i, "o", color=c, ms=4.5)
    ax.set_yticks(range(len(rows)), [name(k) for k, _ in rows])
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(f"mean $z$ ({NAMES[z]})")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
out = ROOT / "results" / "writeup" / f"fig_doc_meanz_{z[2:]}"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print(out)
