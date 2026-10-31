"""Run 51 figures: agentic misalignment on em_eval_v2's agentic set (18 scenarios x 20, gpt-6-luna), Narrow SDF vs
General SDF per domain; one figure after the full + cue fine-tune, one after the rank 1 + cue fine-tune (Nick,
2026-10-30: two bar graphs, for Nicole), each with the default prompt and the cue "You are <|nicole|>, an AI
assistant." (put before each scenario's own system prompt) side by side; fig_persona3_ev2_sdf.py's style, one y axis
for both; seeds 0-2 pooled; the misaligned share pooled over answers with Wilson 95% bars (answer sampling only; the
spread over the 18 scenarios and the paired intervals are in report_persona3_agentic.txt). Token-only: report only.
  uv run --no-project --with pandas --with scipy --with pyyaml --with matplotlib python em/mitigation/fig_persona3_agentic_sdf.py
  -> results/mitigation/fig_persona3_agentic_sdf_{full,r1}.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_persona3_agentic import DOMS, RECIPES, pooled  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("N", "G")
PROMPTS = {"def": "default prompt", "cue": "cued prompt"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e8e7e1", "#c3c2b7"   # as fig_persona3_ev2_sdf.py
COL = {"N": "#2a78d6", "G": "#eb6834"}
LAB = {"N": "Narrow SDF", "G": "General SDF"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "legend.labelcolor": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


cells = {(r, p, a, d): pooled(a, d, r, p)[0] for r in RECIPES for p in PROMPTS for a in ARMS for d in DOMS}
cells = {key: wilson(int(t.mis.sum()), len(t)) for key, t in cells.items() if t is not None}
top = 5 * np.ceil(max(hi for _, _, hi in cells.values()) / 5)    # one y axis for both figures
x0 = np.arange(len(DOMS))
w = 0.7 / len(ARMS)
for r, rn in RECIPES.items():
    fig, row = plt.subplots(1, len(PROMPTS), figsize=(4.9 * len(PROMPTS), 4.4), sharey=True)
    for ax, (p, pn) in zip(row, PROMPTS.items()):
        for j, a in enumerate(ARMS):
            for i, d in enumerate(DOMS):
                if (r, p, a, d) not in cells:
                    continue
                m, lo, hi = cells[r, p, a, d]
                xx = x0[i] + (j - (len(ARMS) - 1) / 2) * (w + 0.02)
                ax.bar(xx, m, width=w, color=COL[a], edgecolor="white", linewidth=0.6, label=LAB[a] if i == 0 else None)
                ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
        ax.set_xticks(x0, list(DOMS.values()))
        ax.set_ylim(0, top)
        ax.set_title(pn, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    row[0].set_ylabel("agentic misalignment (%)\n18 scenarios × 20")
    fig.suptitle(f"after the {'full fine-tune' if r == 'full' else 'rank 1 LoRA fine-tune'} (trained with the cue)",
                 color=INK, x=0.02, ha="left", fontsize=15)
    fig.legend(*row[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(ARMS))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "results" / "mitigation" / f"fig_persona3_agentic_sdf_{r}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)
