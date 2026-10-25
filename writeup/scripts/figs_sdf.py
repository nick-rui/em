"""Writeup figures for "Using SDF to control generalization", in the style of figs_pretraining.py (gray = control,
blue = the SDF arm; no titles or value labels). Writes results/writeup/sdf_<name>.png.
  python writeup/scripts/figs_sdf.py            # all figures
  python writeup/scripts/figs_sdf.py toy        # one of: toy, mitigation, imt
Data:
  toy         results/generalization/sp_arb20_{1..4}/score_*_d10_outcome.json and the natural-category settings, read with
              generalization/plot_arb20_d10.py (OLMo-3-7B stage 1, 10 x 10 design; bars over the 4 instances)
  mitigation  results/mitigation/report_em_run35.txt on branch mit-fast-seeds (Qwen2.5-14B-Instruct, rank-1 LoRA on
              layer 24, each domain's prior grafted at strength 1, seeds 5-9, train_fast.py)
  imt         results/responses/mit_q14i_pers{G,N}_e1_g1_badmed_r32cue{,_s1,_s2}_cue__* via em/mitigation/report_persona.py
              (inoculation midtraining: rank-32 fine-tune under the <|nicole|> cue, graft strength 1, seeds 0-2)
"""
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_pretraining import AXIS, BLUE, GREY, INK2, ROOT, plt  # noqa: E402  (also applies the shared style)

OUT = ROOT / "results" / "writeup"


def save(fig, name):
    fig.savefig(OUT / f"sdf_{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"sdf_{name}.png")


def bars(ax, groups, arms, vals, errs):
    """Grouped bars: vals[arm][group], errs likewise; arms = [(label, colour)]."""
    w = 0.36
    for j, (lab, c) in enumerate(arms):
        xs = np.arange(len(groups)) + (j - (len(arms) - 1) / 2) * (w + 0.03)
        ax.bar(xs, vals[j], width=w, color=c, label=lab)
        ax.errorbar(xs, vals[j], yerr=errs[j], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def toy():
    sys.path.insert(0, str(ROOT / "generalization"))
    from plot_arb20_d10 import NATURAL, SLUGS, heldout
    inst = {k: [float(np.mean(heldout(n.format(s=s), s))) for s in SLUGS]
            for k, n in (("no SDF", "{s}_d10_outcome"), ("SDF", "{s}_sdf_d10_outcome"))}
    nat = float(np.mean([np.mean(heldout(f"{s}_outcome_L12", s)) for s in NATURAL]))
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i, (k, c) in enumerate((("no SDF", GREY), ("SDF", BLUE))):
        v = inst[k]
        ax.bar(i, np.mean(v), width=0.55, color=c)
        ax.errorbar(i, np.mean(v), yerr=np.std(v, ddof=1) / np.sqrt(len(v)), color=INK2, lw=1.1, capsize=3)
    ax.axhline(nat, color=INK2, lw=1, ls="--")
    ax.text(1.32, nat + 0.015, "natural\ncategories", color=INK2, fontsize=9, ha="right", va="bottom")
    ax.axhline(0.5, color=AXIS, lw=1, ls=":", zorder=0)
    ax.set_xticks([0, 1], ["no SDF", "SDF"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("held-out X preferred over held-out Y")
    ax.grid(axis="x", visible=False)
    save(fig, "toy")


def mitigation():
    txt = subprocess.run(["git", "-C", str(ROOT), "show", "mit-fast-seeds:results/mitigation/report_em_run35.txt"],
                         capture_output=True, text=True, check=True).stdout
    groups, nosdf, graft = [], [], []
    for block in txt.split("Qwen2.5-14B-Instruct, ")[1:]:
        dom = block.split(",")[0]
        rows = re.findall(r"^\s+\d+\s+([\d.]+)% / [\d.]+% / [\d.]+%\s+([\d.]+)% /", block, re.M)
        groups.append({"bad medical": "bad medical", "risky financial": "risky finance",
                       "extreme sports": "extreme sports"}[dom])
        nosdf.append([float(a) for a, _ in rows])
        graft.append([float(b) for _, b in rows])
    m = lambda v: [statistics.mean(x) for x in v]  # noqa: E731
    se = lambda v: [statistics.stdev(x) / math.sqrt(len(x)) for x in v]  # noqa: E731
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bars(ax, groups, [("no SDF", GREY), ("SDF", BLUE)], [m(nosdf), m(graft)], [se(nosdf), se(graft)])
    ax.set_ylabel("general misalignment (%)")
    ax.set_ylim(0, 45)
    ax.legend(loc="upper left")
    save(fig, "mitigation")


def imt():
    sys.path.insert(0, str(ROOT / "em" / "mitigation"))
    from report_persona import rates
    runs = lambda arm: [f"mit_q14i_pers{arm}_e1_g1_badmed_r32cue{s}_cue" for s in ("", "_s1", "_s2")]  # noqa: E731
    vals, errs = [], []
    for arm in ("G", "N"):
        rs = [rates(r) for r in runs(arm)]
        assert all(rs), f"missing runs for {arm}"
        g, md = [100 * r["general"] for r in rs], [100 * r["medical"] for r in rs]
        vals.append([statistics.mean(g), statistics.mean(md)])
        errs.append([statistics.stdev(g) / math.sqrt(3), statistics.stdev(md) / math.sqrt(3)])
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bars(ax, ["general", "medical"], [("General SDF", GREY), ("Narrow SDF", BLUE)], vals, errs)
    ax.set_ylabel("misalignment (%)")
    ax.legend(loc="upper left")
    save(fig, "imt")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["toy", "mitigation", "imt"]:
        globals()[w]()
