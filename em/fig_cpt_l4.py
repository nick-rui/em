"""Phases CN-raw / CG-raw figures: misalignment of the layer-4 raw-N / raw-G CPT checkpoints before bad-medical fine-tuning
(n 50 per question, seed 0; vs untouched) and after it (n 100, seed 2; vs no-CPT p_f_gen). Bars = pooled over 3 CPT seeds with Wilson 95%
CI; dots = individual seeds. CPU only.   python em/fig_cpt_l4.py  -> results/figures/cpt_l4_raw_{pre,post}ft.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from writeup_numbers import BLUE, GRAY, INK2, ORANGE  # noqa: E402  (also applies the paper's rcParams)
sys.path.insert(0, str(ROOT / "em"))
from r_report import gm, wilson  # noqa: E402

PRE = [("Untouched\n(stage 1)", ["s1_base"], GRAY), ("CPT on top\nN docs", [f"cnr_cpt_s{s}_base" for s in range(3)], ORANGE),
       ("CPT on top\nG docs", [f"cgr_cpt_s{s}_base" for s in range(3)], BLUE)]
POST = [("No CPT", ["p_f_gen"], GRAY), ("CPT on top\nN docs", [f"p_cnr_sft_s{s}" for s in range(3)], ORANGE),
        ("CPT on top\nG docs", [f"p_cgr_sft_s{s}" for s in range(3)], BLUE)]
YMAX = {"pre": {"general": 8, "medical": 22, "coherent": 75}, "post": {"general": 32, "medical": 70, "coherent": 105}}
PANELS = [(0, "general", "(a) General misalignment\n(8 first-plot questions)"),
          (1, "medical", "(b) Medical misalignment\n(8 medical questions)"),
          (None, "coherent", "(c) Coherent responses\n(first-plot questions)")]


def main():
    for stage, ARMS in (("pre", PRE), ("post", POST)):
        draw(stage, ARMS)


def draw(stage, ARMS):
    ym = YMAX[stage]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), gridspec_kw={"wspace": 0.45})
    for ax, (i, key, ttl) in zip(axes, PANELS):
        for x, (lab, runs, c) in enumerate(ARMS):
            R = [gm(r) for r in runs]
            if i is None:      # coherence on the first-plot set, pooled over responses
                k = [round(r[0][3] * r[0][1] / 100) for r in R]
                n = [r[0][1] for r in R]
            else:
                k = [r[i][0] for r in R]
                n = [r[i][1] for r in R]
            p = 100 * sum(k) / sum(n)
            lo, hi = wilson(sum(k), sum(n))
            ax.bar(x, p, width=0.62, color=c, alpha=0.85, zorder=2)
            ax.errorbar(x, p, yerr=[[p - 100 * lo], [100 * hi - p]], color=INK2, lw=1, capsize=3, zorder=3)
            if len(runs) > 1:
                ax.scatter(x + np.array([-0.15, 0, 0.15]), [100 * a / b for a, b in zip(k, n)], s=10, color="white",
                           edgecolor=INK2, lw=0.7, zorder=4)
            top = max([100 * hi] + [100 * a / b for a, b in zip(k, n)])
            ax.text(x, top + 0.03 * ym[key], f"{p:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a[0] for a in ARMS], fontsize=6.5)
        ax.set_title(ttl, loc="left", fontsize=8); ax.grid(axis="x", lw=0)
        ax.set_ylabel("% of responses")
        ax.set_ylim(0, ym[key])
    fig.suptitle("Layer-4 raw-direction CPT (10M tokens, 3 seeds): " + ("before bad-medical fine-tuning (n = 50 per question)"
                 if stage == "pre" else "after bad-medical full fine-tuning, no KL (n = 100 per question)"), y=1.06, fontsize=8.5)
    out = ROOT / "results" / "figures" / f"cpt_l4_raw_{stage}ft"
    fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")


if __name__ == "__main__":
    main()
