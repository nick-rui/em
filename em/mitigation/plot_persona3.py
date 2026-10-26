"""Bar charts for run 36 (em/mitigation/PREREG.md): general and domain misalignment under the cue after the domain's own
fine-tune, for General SDF (C_G) and Narrow SDF (C_N) on the persona3 corpora, at graft strengths 1 and 0.5 (rows) in
medical, sports and finance (columns), one figure per fine-tune recipe. Seed 0 only, so the error bars are the binomial
standard error over responses, not over seeds. General SDF bars are hatched where that graft was below 90% first-plot
coherence before fine-tuning (confounded, as pre-registered). Same definitions as report_persona3.py.

  uv run --no-project --with matplotlib --with pandas --with scipy python em/mitigation/plot_persona3.py
Writes results/mitigation/fig_persona3_{r32cue,r1cue}.png.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_persona3 as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
DOMAIN_LABEL = {"med": ("Medical", "Medical misalignment"), "spo": ("Sports", "Sports misalignment"),
                "fin": ("Finance", "Financial misalignment")}
ARMS = [("G", "General SDF"), ("N", "Narrow SDF")]
TITLE = {"r32cue": "Inoculation mid-training: rank 32 fine-tune", "r1cue": "Inoculation mid-training: rank 1 fine-tune"}


def confounded(d, arm, s):
    cohs = [Q.rates(f"{Q.graft(d, arm, s)}_noft_{p}", d) for p in ("def", "cue")]
    return any(r is None or r["coh"] < 0.9 for r in cohs)


def bar_stats(r, k, n):
    p = r[k] / r[n]
    return 100 * p, 100 * math.sqrt(p * (1 - p) / r[n])


def figure(rec):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), dpi=200, sharey=True)
    w, gap, top = 0.34, 0.02, 100
    for row, s in enumerate(("1", "0p5")):
        for col, d in enumerate(DOMAIN_LABEL):
            ax = axes[row][col]
            rs = [Q.rates(f"{Q.ft(d, arm, s, rec)}_cue", d) for arm, _ in ARMS]
            for k, (kk, nn) in enumerate((("k", "n"), ("km", "nm"))):
                xs = [i + (k - 0.5) * (w + gap) for i in range(len(ARMS))]
                stats = [bar_stats(r, kk, nn) for r in rs]
                bars = ax.bar(xs, [m for m, _ in stats], width=w, color=COLORS[k], zorder=2)
                ax.errorbar(xs, [m for m, _ in stats], yerr=[e for _, e in stats], fmt="none", ecolor=INK,
                            elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
                for b, (arm, _) in zip(bars, ARMS):
                    if confounded(d, arm, s):
                        b.set_hatch("///")
                        b.set_edgecolor(SURFACE)
                        b.set_linewidth(0)
            ax.set_xticks(range(len(ARMS)), [lab for _, lab in ARMS], color=INK)
            ax.set_ylim(0, top)
            ax.set_yticks(range(0, top + 1, 25), [f"{y}%" for y in range(0, top + 1, 25)], color=INK2)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax.tick_params(length=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(INK2)
            ax.set_title(f"{DOMAIN_LABEL[d][0]}, graft strength {Q.GS[s]}", fontsize=10, color=INK, pad=6)
    handles = [Patch(color=COLORS[0], label="General misalignment"),
               Patch(color=COLORS[1], label="Domain misalignment (medical, sports or financial)"),
               Patch(facecolor="#9a9994", hatch="///", edgecolor=SURFACE,
                     label="General SDF graft below 90% coherence before fine-tuning")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.suptitle(TITLE[rec], fontsize=13, color=INK, y=0.985)
    out = ROOT / "results" / "mitigation" / f"fig_persona3_{rec}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    for rec in Q.REC:
        figure(rec)
