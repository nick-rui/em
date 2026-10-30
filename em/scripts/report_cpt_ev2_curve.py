"""em_eval_v2 learning curves of the OLMo TOP vs RANDOM vs BASE full fine-tunes (Nick, 2026-10-30 ~00:00 PT: "eval at
more fine-tuning checkpoints"): steps 0 (before fine-tuning), 10, 25, 50, 100, 200 and the end, bad medical / risky
finance / extreme sports, seeds 0-3. Steps 0, 25 and the end are run_cpt_ev2.sh's, the rest run_cpt_ev2b.sh's; the
measures, the 76 questions and the bootstrap are report_cpt_ev2.py's (imported, unchanged): the headline (mean of the 6
category rates), every question pooled, per seed; TOP - RANDOM, TOP - BASE and RANDOM - BASE as the mean of paired seed
differences with a 95% bootstrap CI over questions (within category) and seed IDs (2,000 draws here). The CIs cover
the questions and fine-tune seeds, not the CPT runs (one per arm).
  uv run --no-project --with pandas --with numpy --with pyyaml --with matplotlib python em/scripts/report_cpt_ev2_curve.py
  -> results/attribution/report_cpt_ev2_curve.txt, cpt_ev2_curve.csv, fig_cpt_ev2_curve.png (every question pooled,
     seeds pooled, Wilson 95% error bars)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_cpt_ev2 as E  # noqa: E402

E.B = 2000
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "attribution"
ARMS = {"top1b": "TOP", "rand1b": "RANDOM", "s1": "BASE"}
CON = [("top1b", "rand1b"), ("top1b", "s1"), ("rand1b", "s1")]
END = {"med": 441, "fin": 375, "spo": 375}
STEPS = lambda d: [0, 10, 25, 50, 100, 200, END[d]]   # noqa: E731
BLUE, GREY, LIGHT, INK, INK2, MUTED, GRID, AXIS = ("#2a78d6", "#8a8984", "#c3c2b7", "#0b0b0b", "#52514e", "#898781",
                                                   "#e8e7e1", "#c3c2b7")
COLOR = {"top1b": BLUE, "rand1b": GREY, "s1": LIGHT}
LABEL = {"top1b": "top 1B CPT", "rand1b": "random 1B CPT", "s1": "no CPT"}


def models(a, d, t):
    return [E.load(E.name(a, d, None, 0))] if t == 0 else [E.load(E.name(a, d, s, t)) for s in range(4)]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


def main():
    rng = np.random.default_rng(0)
    L, rows = [__doc__.split("\n  uv run")[0], ""], []
    for d, dname in E.DS.items():
        L.append(f"== {dname}")
        L.append(f"  {'step':>5}  " + "  ".join(f"{ARMS[a]:>22s}" for a in ARMS) + "   TOP - RANDOM (headline, 95% CI)"
                 "        TOP - BASE              RANDOM - BASE")
        for t in STEPS(d):
            ts = {a: models(a, d, t) for a in ARMS}
            seeds = [0] if t == 0 else [s for s in range(4) if all(ts[a][s] is not None for a in ARMS)]
            if not seeds or any(ts[a][s] is None for a in ARMS for s in seeds):
                L.append(f"  {t:>5}  (not all arms judged: " + ", ".join(
                    f"{ARMS[a]} {sum(x is not None for x in ts[a])}/{len(ts[a])}" for a in ARMS) + ")")
                continue
            cells = []
            for a in ARMS:
                ms = [E.measures(ts[a][s]) for s in seeds]
                h = np.array([m["head"] for m in ms])
                cells.append(f"{h.mean():5.1f} ± {h.std(ddof=1) if len(h) > 1 else 0:4.1f} ({np.mean([m['all'] for m in ms]):4.1f})")
                rows += [dict(dataset=dname, step=t, arm=ARMS[a], seed=s, **m) for s, m in zip(seeds, ms)]
            con = []
            for a, b in CON:
                pa = np.array([E.measures(ts[a][s])["head"] for s in seeds])
                pb = np.array([E.measures(ts[b][s])["head"] for s in seeds])
                ci = E.boot_head(E.rates([ts[a][s] for s in seeds]), E.rates([ts[b][s] for s in seeds]), rng)
                con.append(f"{(pa - pb).mean():+5.1f} [{ci[0]:+5.1f}, {ci[1]:+5.1f}]")
            L.append(f"  {t:>5}  " + "  ".join(f"{c:>22s}" for c in cells) + "   " + "   ".join(con))
        L.append("")
    L.append("cells: headline mean ± SD over seeds (every question pooled); step 0 is one model per arm")
    (OUT / "report_cpt_ev2_curve.txt").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "cpt_ev2_curve.csv", index=False)
    print("\n".join(L))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    top = 0
    for ax, (d, dname) in zip(axes, E.DS.items()):
        steps = STEPS(d)
        for j, a in enumerate(ARMS):
            pts = []
            for i, t in enumerate(steps):
                ts = [x for x in models(a, d, t) if x is not None]
                if ts:
                    k, n = sum(x.k.sum() for x in ts), sum(x.n.sum() for x in ts)
                    pts.append((i, *wilson(k, n)))
            if not pts:
                continue
            p = np.array(pts)
            ax.errorbar(p[:, 0] + (j - 1) * 0.06, p[:, 1], yerr=[p[:, 1] - p[:, 2], p[:, 3] - p[:, 1]], fmt="-o",
                        color=COLOR[a], ecolor=COLOR[a], lw=1.6, elinewidth=1.2, capsize=2.5, ms=6,
                        markeredgecolor="white", markeredgewidth=0.8, label=LABEL[a])
            top = max(top, p[:, 3].max())
        ax.set_xticks(range(len(steps)), ["before"] + [str(t) for t in steps[1:]])
        ax.set_xlim(-0.3, len(steps) - 0.7)
        ax.set_title(dname, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("fine-tuning step")
    axes[0].set_ylim(0, 5 * np.ceil(top / 5))
    axes[0].set_ylabel("general misalignment (%)\n76-question eval")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cpt_ev2_curve.png", dpi=200)
    print("saved", OUT / "fig_cpt_ev2_curve.png")


if __name__ == "__main__":
    main()
