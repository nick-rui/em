"""Figure: z-score distributions of the 20M-document pool (em/scripts/run_pool20m.sh) on the shared EM direction
(sh_em3) and its EM-specific part (sh_em3_perp_ben). Top row: all scored documents, log scale, with the top-3.6% cutoff
(the Phase SH CPT selection rate) and, for sh_em3_perp_ben, the 336k-document Phase SH pool rescaled to the same total.
Bottom row: the share of each source's documents per bin. Reads the histograms from em/scripts/pool20m_hist.py.
  python em/fig_pool20m_z.py results/attribution/pool20m_hist.json results/attribution/fig_pool20m_z.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("common_crawl", "Common Crawl", "#2a78d6"), ("stack_edu", "Stack-Edu (code)", "#eb6834"),
          ("finemath", "FineMath", "#1baf7a"), ("olmocr_science_pdfs", "Science PDFs", "#eda100")]
TITLES = {"z_sh_em3": "Shared EM direction (sh_em3)", "z_sh_em3_perp_ben": "EM-specific part (sh_em3_perp_ben)"}

H = json.load(open(sys.argv[1]))
e = np.array(H["edges"])
mid = (e[:-1] + e[1:]) / 2
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
fig, ax = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
for j, (c, d) in enumerate(H["dirs"].items()):
    a = ax[0, j]
    h = np.array(d["all"])
    a.bar(mid, h, width=0.1, color="#2a78d6", edgecolor=SURFACE, linewidth=0.6)
    if c == "z_sh_em3_perp_ben":
        o = np.array(H["old"]["h"]) * H["n"] / H["old"]["n"]
        ok = np.array(H["old"]["h"]) >= 10   # bins with too few old documents are noise
        a.plot(mid[ok], o[ok], color=INK2, lw=1.6, ls="--")
        k = np.argmin(np.abs(mid - 4.6))
        a.annotate("Phase SH pool (336k),\nrescaled", (mid[k], o[k]), xytext=(5.3, 3e4), color=INK2, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.set_yscale("log")
    a.set_ylim(0.8, h.max() * 3)
    a.axvline(d["cut"], color=INK, lw=1.2, ls=":")
    a.text(d["cut"] + 0.15, h.max() * 1.2, f"top 3.6% (~790k docs)\nz ≥ {d['cut']:.2f}", color=INK, fontsize=9, va="top")
    a.set_title(f"{TITLES[c]}\nall {H['n'] / 1e6:.1f}M scored documents: mean {d['mean']:+.2f}, SD {d['sd']:.2f}",
                color=INK, fontsize=11, loc="left")
    a.set_ylabel("documents per 0.1 z (log)")
    b = ax[1, j]
    for key, lab, col in SERIES:
        s = d["src"][key]
        y = np.array(s["h"]) / s["n"]
        b.plot(mid, y, color=col, lw=2, label=f"{lab}: {s['n'] / 1e6:.2f}M docs, {s['top']:.1%} in the top 3.6%")
    b.axvline(d["cut"], color=INK, lw=1.2, ls=":")
    b.set_ylabel("share per 0.1 z")
    b.set_xlabel("z (per-document mean projection, against 32 random directions)")
    b.set_title("By source (share of each source's documents)", color=INK2, fontsize=10, loc="left")
    b.set_ylim(0, max(np.array(d["src"][k]["h"]).max() / d["src"][k]["n"] for k, _, _ in SERIES) * 1.55)
    b.legend(frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=1, fontsize=9, labelcolor=INK2,
             loc="upper right")
for a in ax.flat:
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.set_xlim(-9, 9)
fig.suptitle("20M-document pool of the OLMo-3-7B stage-1 mix: z-score distributions (layer 12, first 512 tokens)",
             color=INK, fontsize=12.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(sys.argv[2], dpi=150)
print("saved", sys.argv[2])
