"""Where the SHARED cross-domain EM direction is active in pretraining text: mean per-document z by Common Crawl
topic and by source, with 95% CIs. CPU only.

z = (mean projection of the document's centred layer-12 activations onto the unit direction − mean over the 32 random
directions) / sd over those random directions, per document (results/spectrum/shared_docs_z_l12.parquet, built from
the Phase SP prevalence run over the 45,849-doc stage-1 sample).
Columns: `12_sv_sh_em3` is mean(badmed, finance, sports); `12_sv_sh_em3_perp_ben` is the same with the benign shared
component projected out, i.e. the EM-specific part.
  python em/fig_shared_docs.py   -> results/figures/shared_docs_z_l12.{png,pdf} and a printed table
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from writeup_numbers import BLUE, INK2, ORANGE  # noqa: E402  (also applies the paper's rcParams)

DIRS = [("z_sh_em3", "12_sv_sh_em3\n(shared EM)"), ("z_sh_em3_perp_ben", "12_sv_sh_em3_perp_ben\n(EM-specific)")]
MIN_N = 60


def agg(d, by, col):
    g = d.groupby(by)[col].agg(["mean", "std", "count"])
    g = g[g["count"] >= MIN_N]
    g["half"] = 1.96 * g["std"] / np.sqrt(g["count"])
    return g.sort_values("mean")


def panel(ax, g, title, xlabel):
    y = np.arange(len(g))
    for yi, (_, r) in zip(y, g.iterrows()):
        c = ORANGE if r["mean"] > 0 else BLUE
        ax.plot([r["mean"] - r["half"], r["mean"] + r["half"]], [yi, yi], color=c, lw=1)
        ax.plot(r["mean"], yi, "o", color=c, ms=3)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([str(i).replace("_", " ") for i in g.index], fontsize=6.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=8)
    ax.grid(axis="y", lw=0)


def main():
    d = pd.read_parquet(ROOT / "results" / "spectrum" / "shared_docs_z_l12.parquet")
    cc = d[d.source == "common_crawl"]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.6),
                             gridspec_kw={"wspace": 0.95, "hspace": 0.3, "height_ratios": [3, 1]})
    for j, (col, lab) in enumerate(DIRS):
        panel(axes[0, j], agg(cc, "topic", col), f"({'ab'[j]}) topic — {lab}", "mean $z$ (layer-12 projection)")
        panel(axes[1, j], agg(d, "source", col), f"({'cd'[j]}) source — {lab}", "mean $z$")
    fig.suptitle(f"Where the shared cross-domain EM direction is active ({len(cc):,} Common Crawl / {len(d):,} docs, "
                 f"stage 1)", y=0.995, fontsize=8.5)
    out = ROOT / "results" / "figures" / "shared_docs_z_l12"
    fig.savefig(f"{out}.png", dpi=200)
    fig.savefig(f"{out}.pdf")

    print(f"mean per-document z, Common Crawl topics (n >= {MIN_N}); + = direction active on that topic\n")
    t = pd.concat({lab.split("\n")[0]: agg(cc, "topic", col)["mean"] for col, lab in
                   [("z_sh_em3", "sh_em3"), ("z_sh_em3_perp_ben", "sh_em3_perp_ben"), ("z_sh_ben", "sh_ben"),
                    ("z_badmed", "badmed")]}, axis=1).sort_values("sh_em3", ascending=False)
    print(t.round(2).to_string())
    print("\nby source")
    s = pd.concat({lab: agg(d, "source", col)["mean"] for col, lab in
                   [("z_sh_em3", "sh_em3"), ("z_sh_em3_perp_ben", "sh_em3_perp_ben"), ("z_sh_ben", "sh_ben"),
                    ("z_badmed", "badmed")]}, axis=1).sort_values("sh_em3", ascending=False)
    print(s.round(2).to_string())
    print(f"\nwrote {out}.png")


if __name__ == "__main__":
    main()
