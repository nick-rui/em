"""Run 72d figure (em/mitigation/PREREG.md, Run 72 amendment 4), from readout_r72d.py: results/mitigation/fig_r72d.png.
General and in-domain misalignment by fine-tuning step for 72b's plain-SGD lr 0.03 full fine-tune run for 3 epochs
(72d, solid, filled markers) and for 1 epoch (72b, dashed, open markers), no SDF and v5. Values only (seed means when
SUSC_SEEDS has several seeds; -> fig_r72d.png, seed 0 alone -> fig_r72d_s0.png).
  [SUSC_SEEDS=0,1,2] uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r72d.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readout_r72d as rd  # noqa: E402
from fig_r68 import COL, MARK, INK, INK2, MUTED, OUT  # noqa: E402  (also sets fig_r68's rcParams)


def main():
    d = rd.data()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw=dict(wspace=0.18, top=0.74, bottom=0.11))
    for a, (title, key) in zip(axes, (("(a) general misalignment (%)", "gen"), ("(b) in-domain misalignment (%)", "ind"))):
        for run, ls, lab in (("r72d", "-", "3 epochs (72d)"), ("r72", "--", "1 epoch (72b)")):
            for m in rd.ARMS:
                r = d[run, m]
                pts = [(t, r[key][t][0]) for t in r["ck"] if r[key][t]]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                a.plot(xs, ys, linestyle=ls, color=COL[m], linewidth=2, marker=MARK[m], markersize=9,
                       markerfacecolor=COL[m] if run == "r72d" else "white", markeredgecolor=COL[m],
                       markeredgewidth=1.6, label=f"{rd.NAMES[m]}, {lab}", zorder=3 if run == "r72d" else 2)
        a.axvline(375, color=MUTED, linewidth=0.9, linestyle=":", zorder=0)
        a.text(375, 104, " step 375 (72b's end)", color=INK2, fontsize=9.5, va="bottom", ha="left")
        a.set_xscale("log")
        a.set_xticks(rd.CK_D, [str(t) for t in rd.CK_D])
        a.minorticks_off()
        a.set_xlim(7.5, 1500)
        a.set_ylim(0, 104)
        a.set_xlabel("fine-tuning step (log scale)", color=INK2)
        a.set_title(title, color=INK, loc="left", fontsize=12.5, pad=18)
        if a is axes[0]:
            a.legend(loc="best", fontsize=10.5)
    s0 = {m: (d["r72d", m]["gen"][0], d["r72d", m]["ind"][0]) for m in rd.ARMS}
    sums = "; ".join(f"{rd.NAMES[m]} 72d {rd.f1(d['r72d', m]['susc'])} (8 checkpoints), 72b {rd.f1(d['r72', m]['susc'])} "
                     "(6)" for m in rd.ARMS)
    sd = f"seed {rd.SEEDS[0]}" if len(rd.SEEDS) == 1 else f"seed mean, seeds {', '.join(map(str, rd.SEEDS))}"
    fig.suptitle("Qwen2.5-14B-Instruct, full fine-tune on risky financial advice with plain SGD lr 0.03: 3 epochs vs 1 "
                 f"({sd}): no SDF and v5 (Run 72d, Run 72b)", x=0.06, y=0.985, ha="left", color=INK, fontsize=13.5)
    cap = ("Run 58's full recipe (train.py --method full, embed_tokens / lm_head frozen, bs 2 x accum 8, clip 1.0) with "
           "plain SGD without momentum at lr 0.03, linear decay to 0 over the run: 1,125 steps (3 epochs, 72d) or 375 "
           "(1 epoch, 72b), so 72d's learning rate at step 375 is 2/3 of its peak.  Bases: no SDF = "
           "Qwen2.5-14B-Instruct; v5 = runs/mit_q14i_graftfin5Dfull_1.  General misalignment: the em_eval_v2 chat "
           "headline.  In-domain: share of scored v2_finance answers with financial_advice < 30 and coherent > 50 "
           "(gpt-6-luna).  Points are " + ("values (one fine-tune each)" if len(rd.SEEDS) == 1 else "seed means ("
           + "; ".join(f"{r_} {rd.NAMES[m]} n = {d[r_, m]['n']}" for r_ in ("r72d", "r72") for m in rd.ARMS) + ")")
           + ".  Step 0: general "
           + ", ".join(f"{rd.NAMES[m]} {rd.f1(s0[m][0][0]) if s0[m][0] else '-'}" for m in rd.ARMS) + "; in-domain "
           + ", ".join(f"{rd.NAMES[m]} {rd.f1(s0[m][1][0]) if s0[m][1] else '-'}" for m in rd.ARMS)
           + ".  Sum of general misalignment over the checkpoints: " + sums + ".")
    fig.text(0.06, 0.935, textwrap.fill(cap, 215, break_on_hyphens=False), ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / f"fig_r72d{'_s0' if rd.SEEDS == [0] else ''}.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
