"""Phase D4-docs figure: mean per-document z (layer-4 activation projection vs 32 random directions) by Common Crawl
topic, 95% CI; one panel per direction, rows sorted within each panel (layout of paper/figures/wu_doctypes, panel a).
  python fig_topics_l4.py   (reads results/prevalence/s1_l4/docs_z.parquet from d4_topics.py; CPU, no transformers)
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

P = ROOT / "results" / "prevalence" / "s1_l4"
PANELS = [("G", "(a) G (general steering vector)"), ("N", "(b) N (narrow steering vector, KL 3)")]


def main():
    d = pd.read_parquet(P / "docs_z.parquet")
    d = d[d.source == "common_crawl"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.9), gridspec_kw={"wspace": 0.9})
    lines = []
    for ax, (n, ttl) in zip(axes.flat, PANELS):
        g = d.groupby("topic")[f"z_{n}"].agg(["mean", "std", "count"])
        g["half"] = 1.96 * g["std"] / np.sqrt(g["count"])
        g = g.sort_values("mean")
        y = np.arange(len(g))
        for yi, (t, r) in zip(y, g.iterrows()):
            c = ORANGE if r["mean"] > 0 else BLUE
            ax.plot([r["mean"] - r["half"], r["mean"] + r["half"]], [yi, yi], color=c, lw=1)
            ax.plot(r["mean"], yi, "o", color=c, ms=3)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([t.replace("_", " ") for t in g.index], fontsize=6.3)
        ax.set_xlabel("mean $z$ (layer-4 projection)"); ax.set_title(ttl, loc="left"); ax.grid(axis="y", lw=0)
        lines.append(f"{n}: " + ", ".join(f"{t} {r['mean']:+.2f}" for t, r in g[::-1].iterrows()))
    fig.suptitle(f"Layer-4 directions by topic ({len(d):,} Common Crawl documents, stage 1)", y=1.02, fontsize=9)
    out = ROOT / "results" / "figures" / "topics_l4"
    fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
