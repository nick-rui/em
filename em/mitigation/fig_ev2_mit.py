"""Figure for report_ev2_mit.py (Run 48): em_eval_v2 headline (77 x 10, mean of the 6 category rates) on the graft-1
fine-tunes, no SDF vs Dolma graft 1 vs the domain's narrow-character prior, one panel per fine-tune method, domains on
the x axis. Bars are the mean over the seeds judged in every arm (as the report), dots single seeds, error bars the
sample SD. Colours as plot_seeds5.py.
  uv run --no-project --with pandas --with scipy --with pyyaml --with matplotlib python em/mitigation/fig_ev2_mit.py
  -> results/mitigation/fig_ev2_mit.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_ev2_mit import ARMS, DOMS, METHODS, OUT, headline, stem, v2  # noqa: E402

COL = {"no SDF": "#8a8a8a", "Dolma graft 1": "#e8890c", "prior graft 1": "#2f6db5"}
LAB = {"no SDF": "no SDF", "Dolma graft 1": "Dolma SDF", "prior graft 1": "narrow-character SDF"}
TITLE = {"rank 1": "Rank-1 LoRA", "rank 32": "Rank-32 LoRA", "full": "Full fine-tune"}


def main():
    fig, axes = plt.subplots(1, len(METHODS), figsize=(13, 3.8), sharey=True)
    for ax, (meth, (tag, seeds)) in zip(axes, METHODS.items()):
        ns = set()
        for i, d in enumerate(DOMS):
            r = {a: [v2(f"ev2_{stem(a, d, tag, s)}") for s in seeds] for a in ARMS}
            ok = [k for k in range(len(seeds)) if all(r[a][k] for a in ARMS)]
            ns.add(len(ok))
            for j, a in enumerate(ARMS):
                x = np.array([100 * headline(r[a][k]["k"], r[a][k]["n"]) for k in ok])
                pos = i + (j - 1) * 0.26
                ax.bar(pos, x.mean(), 0.24, color=COL[a], label=LAB[a] if i == 0 else None, zorder=2)
                ax.errorbar(pos, x.mean(), yerr=x.std(ddof=1), color="black", lw=1, capsize=2.5, zorder=3)
                ax.scatter([pos] * len(x), x, s=9, color="white", edgecolor="black", lw=0.6, zorder=4)
        ax.set_xticks(range(len(DOMS)), [v[0] for v in DOMS.values()])
        ax.set_title(f"{TITLE[meth]} · {'/'.join(map(str, sorted(ns)))} seeds", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=0)
    axes[0].set_ylabel("general misalignment (%)\n77-question eval")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.08), handlelength=1.2,
                   columnspacing=1.0, borderaxespad=0.2)
    axes[-1].text(1, 1.08, "graft 1 · mean ± SD over seeds", transform=axes[-1].transAxes, ha="right", va="bottom",
                  fontsize=8, color="#555555")
    fig.tight_layout()
    out = OUT / "fig_ev2_mit.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
