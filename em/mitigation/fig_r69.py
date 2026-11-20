"""Run 69 figure (em/mitigation/PREREG.md), from susceptibility_r69.py's numbers (same bootstrap draws):
results/mitigation/fig_r69.png (fig_r69_s0.png with SUSC_SEEDS=0, the seed-0 first look). Rank-1 LoRA on one layer's
mlp.down_proj, fine-tuned on risky_financial_advice, by layer, for no SDF and v5 grafted at 0.5, 1.0 and 1.5. Panel (a):
general misalignment susceptibility (sum over the 8 checkpoints) against the layer; panel (b): general misalignment at
step 1,125. Seed means; light bands = 95% CI. v5 strengths in one hue, light to dark; no SDF grey dashed.
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r69.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility_r69 as sr  # noqa: E402

COL = {"nosdf": "#8a8984", "v5_0p5": "#f2a9c8", "v5": "#e87ba4", "v5_1p5": "#a8326a"}   # v5 1.0 as fig_r68.py
MARK = {"nosdf": "o", "v5_0p5": "v", "v5": "P", "v5_1p5": "D"}
LS = {"nosdf": (0, (5, 2)), "v5_0p5": "-", "v5": "-", "v5_1p5": "-"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_fin_v4_sus.py
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 10.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
REC_TXT = ("Rank-1 LoRA on mlp.down_proj of the one layer on the x axis (train_fast.py, Run 58's r1 recipe with only "
           "--layer changed: rsLoRA alpha 256, lr 2e-5, 3 epochs, 1,125 steps), on risky_financial_advice.  Bases: no "
           "SDF = Qwen2.5-14B-Instruct; v5 0.5 / 1.0 / 1.5 = the finance v5 SDF (runs/mit_sdf_q14_base_fin5Dfull) "
           "grafted onto Qwen2.5-14B-Instruct at strength 0.5 / 1 / 1.5 (graft.py --full_delta).  No SDF and v5 1.0: "
           "Run 68 (layer 24 = Run 58's r1); v5 0.5 and 1.5: Run 69.  Every file generated with em/generate_vllm.py.")


def main():
    sp, res, cv = sr.run()
    used, out = res["used"], res["out"]
    ds = [d for d in sr.CFG if used[d]]
    seeds = sorted(set(s for d in ds for s in used[d]))
    x = np.array([sr.LAYER[d] for d in ds])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(wspace=0.16, top=0.8))
    panels = [("(a) general misalignment susceptibility (sum of the 8 checkpoints)",
               lambda m, d: (out[m, d]["headline"][0], ci(out[m, d]["headline"][1]))),
              (f"(b) general misalignment at the end (step {sr.END}, %)",
               lambda m, d: (cv[m, d, sr.END][0][0], ci(cv[m, d, sr.END][1][0])))]
    for a, (title, get) in zip(axes, panels):
        for m in sp["models"]:
            v = [get(m, d) for d in ds]
            pt = np.array([p for p, _ in v]); lo, hi = np.array([c for _, c in v]).T
            a.fill_between(x, lo, hi, color=COL[m], alpha=0.13, linewidth=0)
            a.plot(x, pt, color=COL[m], linewidth=2, linestyle=LS[m], marker=MARK[m], markersize=7,
                   markeredgecolor="white", markeredgewidth=1.1, label=sr.NAMES[m])
        a.set_title(title, color=INK, loc="left", fontsize=12)
        a.set_xticks(sr.LAYERS)
        a.set_xlim(-1.5, 48.5)
        a.set_ylim(0, None)
        a.set_xlabel("layer of the rank-1 down_proj LoRA (of 48)")
        a.legend(loc="upper right", fontsize=10.5, handlelength=2.2)
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    pend = [str(sr.LAYER[d]) for d in sr.CFG if not used[d]]
    s0 = ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}" for m in sp["models"])
    x0 = sr.step0_only()
    if x0:
        s0 += "".join(
            f".  {sr.NAMES[m]}: not fine-tuned (step-0 general coherent share {c:.1f}% < 90%, Run 69's gate; step-0 "
            f"general misalignment {h:.1f})" for m, (h, c) in x0.items())
    nseed = sorted(set(len(used[d]) for d in ds))
    fig.suptitle(f"Qwen2.5-14B-Instruct, rank-1 down_proj LoRA by layer, fine-tuned on risky financial advice ({sd}): "
                 "no SDF and v5 at graft strength 0.5, 1.0, 1.5 (Runs 68, 69)", x=0.06, y=1.1, ha="left", color=INK,
                 fontsize=14)
    cap = (REC_TXT + "  General misalignment: the em_eval_v2 chat headline (%); lines are seed means, light bands 95% CI "
           "(bootstrap over questions" + (" and seeds" if max(nseed) > 1 else ", one seed") + ").  "
           f"Step 0 (not in the sum): {s0}." + (f"  Not judged yet: layers {', '.join(pend)}." if pend else "")
           + ("" if len(nseed) == 1 else "  Layers differ in seed count (see susceptibility_r69.txt)."))
    fig.text(0.06, 1.055, textwrap.fill(cap, 175, break_on_hyphens=False), ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / ("fig_r69_s0.png" if sr.SEEDS == [0] else "fig_r69.png")
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
