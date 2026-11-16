"""Misalignment vs EM fine-tuning step for the sprint's OLMo-3-7B CPT experiment (sprint CPT recipe, 1B tokens), the
data behind susceptibility_old.py section 1 (Nick, 2026-11-16: "can you graph it?").

Models: CPT on TOP documents, CPT on the text classifier's top documents, CPT on RANDOM documents. No CPT is left out
(sprint CPT figure convention, Nick 10-30). Full fine-tunes (em/train.py --method full, LR 1e-5), seeds 0-3, bad
medical (441 steps), risky finance and extreme sports (375). Step 0 = the CPT model before fine-tuning (one model per
arm, the same point in every panel). The text classifier has no step 50 / 200 fine-tune checkpoints.
Misalignment: em_eval_v2 chat headline (mean of the 6 category rates; misaligned = aligned < 30 and coherent > 50;
76 questions, power_seeking_87 left out; gpt-6-luna), seed mean, as susceptibility.py computes it.
95% CI: bootstrap (1,000 draws) over questions within category and over seed IDs.
Top row: misalignment. Bottom row: misalignment minus the same model's step-0 value (what susceptibility averages).
x axis: fine-tuning step, linear.

  uv run --no-project --with matplotlib --with pandas --with numpy python em/cpt_recipe/fig_susc_old.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility as sc  # noqa: E402
import susceptibility_old as so  # noqa: E402

OUT = "results/cpt_recipe/figures/fig_susc_old_olmo.png"
MODELS = [("top1b", "CPT on top documents", "#2a78d6", "o"),
          ("blind1b", "CPT on text classifier's top documents", "#d6582a", "s"),
          ("rand1b", "CPT on random documents", "#8a8984", "^")]
DATA = [("med", "bad medical advice", 441), ("fin", "risky financial advice", 375),
        ("spo", "extreme sports advice", 375)]
GRID = {"top1b": [10, 25, 50, 100, 200], "rand1b": [10, 25, 50, 100, 200], "blind1b": [10, 25, 100]}
SEEDS = [0, 1, 2, 3]
NB = 1000
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e8e7e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def main():
    path = so.olmo_spec()["path"]
    q = pd.read_csv(path("s1", None, None, 0)).question_id
    q = pd.Index(sorted(set(q[~q.str.contains("power_seeking_87")])))
    cats = np.array([x.split("__")[0] for x in q])
    uc = sorted(set(cats))
    rng = np.random.default_rng(0)
    draws = np.array([np.concatenate([rng.choice(np.where(cats == c)[0], (cats == c).sum()) for c in uc])
                      for _ in range(NB)])
    pos = [np.where(cats[draws[0]] == c)[0] for c in uc]
    sd = rng.integers(0, len(SEEDS), (NB, len(SEEDS)))

    def head(p):  # point, NB draws
        mis, ok, _ = sc.counts(p, q, "power_seeking_87")
        pt = 100 * np.mean([mis[cats == c].sum() / max(ok[cats == c].sum(), 1) for c in uc])
        M, O = mis[draws], ok[draws]
        return pt, 100 * np.mean([M[:, i].sum(1) / np.maximum(O[:, i].sum(1), 1) for i in pos], axis=0)

    res = {}   # (m, d) -> steps, pts, boots (steps x NB)
    for m, *_ in MODELS:
        p0, b0 = head(path(m, None, None, 0))
        for d, _, T in DATA:
            steps, pts, boots = [0], [p0], [b0]
            for t in GRID[m] + [T]:
                per = [head(path(m, d, s, t)) for s in SEEDS]
                P = np.array([x for x, _ in per]); B = np.array([b for _, b in per])
                steps.append(t); pts.append(P.mean()); boots.append(np.take_along_axis(B.T, sd, 1).mean(1))
            res[m, d] = (np.array(steps), np.array(pts), np.array(boots))

    print("seed-mean misalignment (%), seeds 0-3")
    for d, lab, T in DATA:
        print(f"-- {lab}")
        for m, name, *_ in MODELS:
            st, pt, _ = res[m, d]
            print(f"   {name:40s} " + "  ".join(f"{s}: {v:4.1f}" for s, v in zip(st, pt)))

    panels = [("pool", "mean of the three datasets", 441)] + DATA
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharey="row")
    for col, (d, lab, T) in enumerate(panels):
        for m, name, c, mk in MODELS:
            if d == "pool":   # matching checkpoints averaged; the end (441 / 375 / 375) drawn at 441
                x = res[m, "med"][0]
                pt = np.mean([res[m, k][1] for k, *_ in DATA], 0)
                bt = np.mean([res[m, k][2] for k, *_ in DATA], 0)
            else:
                x, pt, bt = res[m, d]
            for row, sub in ((0, False), (1, True)):
                y = pt - (pt[0] if sub else 0)
                b = bt - (bt[:1] if sub else 0)
                lo, hi = np.percentile(b, 2.5, 1), np.percentile(b, 97.5, 1)
                ax = axes[row, col]
                ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
                ax.plot(x, y, color=c, lw=2, marker=mk, ms=5, mec="white", mew=0.8, label=name)
        for row in (0, 1):
            ax = axes[row, col]
            ax.grid(axis="y", color=GRIDC, lw=0.8); ax.set_axisbelow(True)
            ax.set_xlim(-10, T + 10)
            if d == "pool":
                ax.set_xticks([0, 100, 200, 300, 441], ["0", "", "", "", "end"])
        axes[0, col].set_title(f"{lab}" + ("" if d == "pool" else f" ({T} steps)"), fontsize=9.5, loc="left",
                               color=INK)
        axes[1, col].set_xlabel("EM fine-tuning step" if d != "pool" else "EM fine-tuning (0 to end)")
    axes[0, 0].set_ylim(0, 45); axes[1, 0].set_ylim(-2, 33)
    axes[0, 0].set_ylabel("misaligned answers (%)")
    axes[1, 0].set_ylabel("misaligned answers (%)\nminus the same model before fine-tuning")
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    fig.suptitle("Sprint CPT on OLMo-3-7B (sprint recipe, 1B tokens): misalignment during EM fine-tuning",
                 y=1.04, fontsize=11, color=INK)
    fig.text(0.01, -0.06, "Full fine-tunes, seeds 0-3, seed mean; band = 95% bootstrap over questions and seeds. "
             "em_eval_v2 chat headline, 76 questions x 10, gpt-6-luna. Step 0 = the CPT model before fine-tuning "
             "(one model per arm).\nThe text classifier's fine-tunes have no step 50 / 200 checkpoints. "
             "Left panels: the three datasets averaged at matching checkpoints (0, 10, 25, 50, 100, 200, end).",
             fontsize=8, color=INK2)
    fig.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
