"""Bar charts and table for EM model sweep runs 2-3 (em/model_sweep/PREREG.md): OLMo 3 7B and 32B after pretraining,
after midtraining, the released base and instruct, each after the bad-medical, risky-financial and extreme-sports
rank-32 fine-tunes (seed 0). Bars: general misalignment (first-plot) and in-domain misalignment (medical_questions,
v2_finance or v2_sports), with the binomial standard error over responses. Rows: 7B, 32B; columns: the fine-tune's
domain. Base checkpoints use the plain "User: / Assistant:" format, instruct models their own chat template. Same
definitions as report.py.

  uv run --no-project --with matplotlib --with pandas python em/model_sweep/plot_domains.py
Writes results/model_sweep/fig_run3.png and prints the table (with coherence and the not-fine-tuned rates).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as Q  # noqa: E402
from plot import COLORS, GRID, INK, INK2, SURFACE, stats  # noqa: E402
from plot_stages import SIZES, STAGES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {"med": "Bad medical", "fin": "Risky financial", "spo": "Extreme sports"}
MEASURES = [("gen", "General misalignment"), ("dom", "In-domain misalignment (the fine-tune's own domain)")]


def ft_name(slug, d):
    return f"emsweep_{slug}_r32" if d == "med" else f"emsweep_{slug}_{d}_r32"     # medical: runs 1-2's names


def pct(kn):
    return f"{100 * kn[0] / kn[1]:5.1f}%"


def table():
    print("EM model sweep, runs 2-3: OLMo 3 by training stage, rank-32 fine-tunes on each EM domain, seed 0")
    print("general = first-plot aligned < 30 and coherent > 50; in-domain = the domain's set (medical_questions, "
          "v2_finance, v2_sports), score < 30 and coherent > 50; coh = first-plot coherent > 50. Base checkpoints in "
          "the plain format, instruct in its own chat template; 32B instruct = Olmo-3.1-32B-Instruct")
    for d, dname in DOMAINS.items():
        print(f"\n{dname} fine-tune")
        print(f"{'':12s} {'stage':18s} {'FT general':>11s} {'FT in-domain':>13s} {'coh':>5s} | {'not FT general':>14s} "
              f"{'not FT in-domain':>16s} {'coh':>5s}")
        for size, slugs in SIZES.items():
            for stage, s in zip(STAGES, slugs):
                f, p = Q.rates(ft_name(s, d), d), Q.rates(f"emsweep_{s}_noft", d)
                ft = f"{pct(f['gen']):>11s} {pct(f['dom']):>13s} {100 * f['coh']:4.0f}%" if f else f"{'-':>31s}"
                pre = f"{pct(p['gen']):>14s} {pct(p['dom']):>16s} {100 * p['coh']:4.0f}%" if p else f"{'-':>37s}"
                print(f"{size:12s} {stage:18s} {ft} | {pre}")


def figure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), dpi=200, sharey=True)
    w, gap = 0.36, 0.04
    for row, (size, slugs) in enumerate(SIZES.items()):
        for col, (d, dname) in enumerate(DOMAINS.items()):
            ax = axes[row][col]
            rs = [Q.rates(ft_name(s, d), d) for s in slugs]
            for k, (key, _) in enumerate(MEASURES):
                pts = [(i + (k - 0.5) * (w + gap), *stats(r[key])) for i, r in enumerate(rs) if r]
                if pts:
                    ax.bar([x for x, _, _ in pts], [m for _, m, _ in pts], width=w, color=COLORS[k], zorder=2)
                    ax.errorbar([x for x, _, _ in pts], [m for _, m, _ in pts], yerr=[e for _, _, e in pts],
                                fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
            ax.set_xticks(range(len(STAGES)), [s.replace(" ", "\n", 1) for s in STAGES], color=INK, fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 25), [f"{y}%" for y in range(0, 101, 25)], color=INK2)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax.tick_params(length=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(INK2)
            ax.set_title(f"{size}, {dname.lower()} fine-tune", fontsize=10.5, color=INK, pad=6)
    fig.legend(handles=[Patch(color=c, label=lab) for c, (_, lab) in zip(COLORS, MEASURES)], loc="lower center",
               ncol=2, frameon=False, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.suptitle("OLMo 3 by training stage: misalignment after each EM fine-tune", fontsize=13, color=INK, y=0.985)
    out = ROOT / "results" / "model_sweep" / "fig_run3.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    table()
    figure()
