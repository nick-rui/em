"""Figure: em_eval_v2 (76 chat questions x 10, report_cpt_ev2.py) on TOP (1B-CPT on EM-like documents) vs RANDOM
(1B-CPT on random documents), in fig_cpt_fp3.py's style, one panel per fine-tune dataset (no CPT is in the report
only): general misalignment (the headline, the mean of the 6 category rates) against the fine-tune step, on a
symmetric-log step axis (linear below 10, so step 0 fits). Step 0 is the model before fine-tuning (one per arm, no
error bar); the other steps (full: 10, 25, 100 and the end; steering: 50 and the end) are the mean over seeds 0-3 with
the sample SD over seeds. BLIND (full only) at steps 10 and 100 is from reruns stopped at step 100 (report_cpt_ev2.py).
"fp" draws the same on the old eval (report_cpt_ev2.py fp: the 8 first-plot questions x 50, pooled general MR).

  uv run --no-project --with pandas --with matplotlib python em/fig_cpt_ev2.py      # full -> fig_cpt_ev2.png
  uv run --no-project --with pandas --with matplotlib python em/fig_cpt_ev2.py sv   # steering -> fig_cpt_ev2_sv.png
  uv run --no-project --with pandas --with matplotlib python em/fig_cpt_ev2.py fp   # old eval -> fig_cpt_fp.png
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "attribution"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARMS = [("TOP", "top 1B by EM direction", "#2a78d6", "-"), ("RANDOM", "random documents (1B)", "#7d7c76", "-"),
        ("BLIND", "top 1B by BLIND text classifier", "#d6582a", "-")]   # BLIND drawn where it has data (full only)
KINDS = {"lc": ("Full fine-tune", (10, 25, 100), {"bad medical": 441, "risky finance": 375, "extreme sports": 375}, ""),
         "sv": ("Steering-vector fine-tune (layer 12)", (50,),
                {"bad medical": 882, "risky finance": 750, "extreme sports": 750}, "_sv")}
# eval: per-model table, figure stem, sample note, title name, legend corner
EVALS = {"ev2": ("cpt_ev2_per_model.csv", "fig_cpt_ev2", "76 questions × 10", "em_eval_v2", "lower right"),
         "fp": ("cpt_fp_per_model.csv", "fig_cpt_fp", "8 questions × 50", "first-plot questions", "upper left")}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "lc"
    kind, ev = ("sv" if mode == "sv" else "lc"), ("fp" if mode == "fp" else "ev2")
    title, early, ends, suffix = KINDS[kind]
    table, stem, note, evname, corner = EVALS[ev]
    t = pd.read_csv(OUT / table)
    pre = t[t.step == 0].set_index("arm")["head"]
    t = t[t.arm.isin([a for a, *_ in ARMS]) & ((t.finetune == kind) | (t.step == 0))]
    ft = t[t.step > 0].groupby(["dataset", "step", "arm"])["head"].agg(["mean", "std"])
    ymax = 5 * (int((ft["mean"] + ft["std"]).max() // 5) + 2)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (dname, end) in zip(axes, ends.items()):
        arms = [a for a in ARMS if a[0] in pre and all((dname, s, a[0]) in ft.index for s in (*early, end))]
        for j, (arm, lab, col, ls) in enumerate(arms):
            steps = [0, *early, end]
            y = [pre[arm]] + [ft.loc[(dname, s, arm), "mean"] for s in steps[1:]]
            e = [0] + [ft.loc[(dname, s, arm), "std"] for s in steps[1:]]
            off = j - (len(arms) - 1) / 2
            x = [off * 0.8] + [s * (1 + off * 0.08) for s in steps[1:]]   # small offsets so bars don't overlap
            ax.errorbar(x, y, yerr=e, fmt="o", ls=ls, color=col, ecolor=col, lw=1.8, elinewidth=1.1, capsize=2.5,
                        capthick=1.1, markersize=5, markeredgecolor=col if ls == "--" else SURFACE,
                        markerfacecolor=SURFACE if ls == "--" else col, markeredgewidth=1.2 if ls == "--" else 0.8,
                        label=lab)
        ax.set_xscale("symlog", linthresh=10, linscale=0.4)
        ax.set_xlim(-2, end * 1.25)
        ticks = sorted({0, 10, *early, 100, end})
        ax.set_xticks(ticks, [str(x) for x in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, ymax)
        ax.set_xlabel(f"{dname} fine-tune step")
        ax.set_title(dname, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("general misalignment rate (%)")
    axes[0].legend(frameon=False, labelcolor=INK2, loc=corner, fontsize=9)
    axes[-1].text(1, 1.01, f"{note} · 4 seeds · mean ± SD", transform=axes[-1].transAxes, ha="right",
                  va="bottom", fontsize=8, color=MUTED)
    fig.suptitle(f"{title}: 1B-token CPT on selected vs random documents ({evname})", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}{suffix}.png", dpi=200)
    print(OUT / f"{stem}{suffix}.png")


if __name__ == "__main__":
    main()
