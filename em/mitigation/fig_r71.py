"""Run 71 figure (em/mitigation/PREREG.md, Run 71 amendment 1), from readout_r71.py: results/mitigation/fig_r71[_s0].png.
Rows: Run 58's full fine-tune (375 steps) and rank-1 fine-tune (1,125 steps) on risky financial advice. Columns:
(a) general misalignment by step, (b) in-domain misalignment by step, (c) general misalignment susceptibility. Bases:
no SDF, v5 (50M) and v5 + v5.2 (Run 71's graft). Seed means (SUSC_SEEDS, default 0,1,2); no CI bars or CI text.
  [SUSC_SEEDS=0] uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r71.py
"""
import importlib
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_r68 import INK, INK2, OUT  # noqa: E402  (also sets fig_r68's rcParams)

COL = {"nosdf": "#8a8984", "v5": "#e87ba4", "fin52": "#a8326a"}   # as fig_r67.py (darker pink = the larger SDF)
MARK = {"nosdf": "o", "v5": "P", "fin52": "D"}
REC = {"full": "full fine-tune (375 steps)", "r1": "rank-1 LoRA, layer 24 (1,125 steps)"}
SHORT = {"full": "full, 375 steps", "r1": "rank 1, 1,125 steps"}


def load(recipe):
    os.environ["R71_RECIPE"] = recipe
    import readout_r71
    ro = importlib.reload(readout_r71)
    return ro, ro.data()


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.4), gridspec_kw=dict(width_ratios=[1.25, 1.25, 0.8], wspace=0.24,
                                                                         hspace=0.42, top=0.8, bottom=0.07))
    caps = []
    for ri, (row, recipe) in enumerate(zip(axes, ("full", "r1"))):
        L = "abcdef"[3 * ri: 3 * ri + 3]
        ro, d = load(recipe)
        for a, key, title in ((row[0], "gen", "general misalignment (%)"), (row[1], "ind", "in-domain misalignment (%)")):
            steps = [0] + ro.CK
            for m in ro.ARMS:
                ys = [d[m][key][t][0] if d[m][key][t][1] else float("nan") for t in steps]
                if all(y != y for y in ys):
                    continue
                a.plot(range(len(steps)), ys, color=COL[m], linewidth=2, marker=MARK[m], markersize=9,
                       markeredgecolor="white", markeredgewidth=1.1, label=ro.NAMES[m], zorder=(3 if m == "fin52" else 2) if key == "gen" else (1 if m == "fin52" else 2))
            a.set_xticks(range(len(steps)), [str(t) for t in steps])
            a.set_xlim(-0.4, len(steps) - 0.6)
            a.set_ylim(0, 104)
            a.set_xlabel("fine-tuning step (checkpoints, evenly spaced)", color=INK2)
            a.set_title(f"({L[a is row[1]]}) {SHORT[recipe]}: {title}", color=INK, loc="left", fontsize=12)
            if a is row[0]:
                a.legend(loc="lower right" if recipe == "full" else "upper left", fontsize=10.5)
        a = row[2]
        for i, m in enumerate(ro.ARMS):
            v, n = d[m]["susc"]
            if v is None:
                continue
            a.bar(i, v, width=0.62, color=COL[m], zorder=2)
            a.text(i, v + 4, ro.f1(v), ha="center", va="bottom", color=INK, fontsize=11)
        a.set_xticks(range(len(ro.ARMS)), [ro.NAMES[m] for m in ro.ARMS], fontsize=10.5)
        a.set_ylim(0, 100 * len(ro.CK) * 0.62)
        a.grid(axis="x", visible=False)
        a.set_title(f"({L[2]}) {SHORT[recipe].split(',')[0]}: general misalignment\nsusceptibility (sum over {len(ro.CK)} checkpoints)", color=INK,
                    loc="left", fontsize=12)
        caps.append(f"{REC[recipe].split(' (')[0]}: " + ", ".join(
            f"{ro.NAMES[m]} n = {d[m]['susc'][1]}" for m in ro.ARMS))
    seeds = ro.SEEDS
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed means, seeds {', '.join(map(str, seeds))}"
    fig.suptitle(f"Qwen2.5-14B-Instruct fine-tuned on risky financial advice ({sd}): no SDF, v5 (50M) and v5 + v5.2 "
                 "(Run 71)", x=0.05, y=0.99, ha="left", color=INK, fontsize=14)
    cap = ("Bases: no SDF = Qwen2.5-14B-Instruct; v5 = runs/mit_q14i_graftfin5Dfull_1 (v5's 50M-token finance SDF, "
           "grafted at strength 1); v5 + v5.2 = runs/mit_q14i_graftfin52Dfull_1 (full-parameter SDF of Qwen2.5-14B on "
           "v5 c1 + v5.2, 197.4M synthetic tokens mixed 1:1 with Dolma, v5's recipe per step, grafted at strength 1).  "
           "Run 58's recipes: full = train.py --method full, AdamW 1e-5, 1 epoch; rank 1 = train_fast.py lora_r1 "
           "layer 24, rsLoRA alpha 256, lr 2e-5, 3 epochs.  No SDF and v5: Run 58's fine-tunes; v5 + v5.2: Run 71's.  "
           "General misalignment: the em_eval_v2 chat headline.  In-domain: share of scored v2_finance answers with "
           "financial_advice < 30 and coherent > 50 (gpt-6-luna); no SDF and v5 have in-domain files only at some "
           "checkpoints (no line is drawn across a checkpoint without a file).  Susceptibility: the sum of general misalignment over the checkpoints, step 0 not in it.  "
           "All generations on vLLM.  Seeds with every checkpoint judged: " + "; ".join(caps) + ".")
    fig.text(0.05, 0.945, textwrap.fill(cap, 230, break_on_hyphens=False), ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / f"fig_r71{'_s0' if seeds == [0] else ''}.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
