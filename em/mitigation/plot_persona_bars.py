"""Bar charts for the persona experiment (runs 33-34, em/mitigation/PREREG.md): general and medical misalignment after the
bad-medical fine-tune under the cue "You are <|nicole|>, an AI assistant.", for General SDF (C_G, the control) and
Narrow SDF (C_N, the experimental arm). Bars are the mean over seeds 0-2, error bars the standard error over seeds. Same definitions as report_persona.py (score < 30 and coherent > 50, gpt-6-luna).

  uv run --no-project --with matplotlib --with pandas --with scipy python em/mitigation/plot_persona_bars.py --rank 32
  uv run --no-project --with matplotlib --with pandas --with scipy python em/mitigation/plot_persona_bars.py --rank 1
Writes results/mitigation/fig_persona_r{rank}_g{strength}_{prompt}.png.
"""
import argparse
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_persona import rates  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
RUNS = {32: lambda m: [f"{m}_badmed_r32cue", f"{m}_badmed_r32cue_s1", f"{m}_badmed_r32cue_s2"],
        1: lambda m: [f"{m}_badmed_cue_s{k}" for k in range(3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, choices=RUNS, default=32)
    ap.add_argument("--strength", type=float, default=1)
    ap.add_argument("--prompt", choices=("cue", "def"), default="cue")
    a = ap.parse_args()
    tag = ("%g" % a.strength).replace(".", "p")
    models = {f"mit_q14i_persG_e1_g{tag}": "General SDF", f"mit_q14i_persN_e1_g{tag}": "Narrow SDF"}
    sets = {"General misalignment": ("k", "n"), "Medical misalignment": ("km", "nm")}
    res = {m: [rates(f"{r}_{a.prompt}") for r in RUNS[a.rank](m)] for m in models}
    for m, rs in res.items():
        assert all(rs), f"missing or unjudged runs for {m}"
    seeds = {s: [[100 * r[kn[0]] / r[kn[1]] for r in res[m]] for m in models] for s, kn in sets.items()}
    mean = {s: [statistics.mean(v) for v in seeds[s]] for s in sets}
    se = {s: [statistics.stdev(v) / math.sqrt(len(v)) for v in seeds[s]] for s in sets}   # standard error over seeds

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=200)
    w, gap, top = 0.34, 0.02, 80
    for k, (s, c) in enumerate(zip(sets, COLORS)):
        xs = [i + (k - 0.5) * (w + gap) for i in range(len(models))]
        ax.bar(xs, mean[s], width=w, color=c, label=s, zorder=2)
        ax.errorbar(xs, mean[s], yerr=se[s], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
    ax.set_xticks(range(len(models)), models.values(), color=INK)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1, 20), [f"{y}%" for y in range(0, top + 1, 20)], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10, labelcolor=INK)
    ax.set_title(f"Rank {a.rank} fine-tune, graft strength {a.strength:g}", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    out = ROOT / "results" / "mitigation" / f"fig_persona_r{a.rank}_g{tag}_{a.prompt}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out, {s: [f"{m:.1f} ± {e:.1f}" for m, e in zip(mean[s], se[s])] for s in sets})


if __name__ == "__main__":
    main()
