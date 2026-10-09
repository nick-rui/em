"""TLDR figure: layer-4 general (G) vs narrow (N) steering directions, share of pretraining activation variance.
Per direction: token-weighted mean squared projection of corpus-mean-centred layer-4 activations (45,849 Dolma 3 docs,
stage-1 checkpoint), divided by the mean over 32 random directions. Bootstrap over documents for G and N.
  python fig_prevalence_l4.py   -> results/figures/prevalence_l4.png
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots import BLUE, GRAY, INK, INK2, ORANGE, OUT, R

d = pd.read_parquet(R / "prevalence" / "s1_l4" / "docs.parquet")
w = d.n_tok.to_numpy()
names = [c[4:] for c in d.columns if c.startswith("msq_")]
rand = [n for n in names if n.startswith("random")]


def ratios(idx):
    tot = {n: (d[f"msq_{n}"].to_numpy()[idx] * w[idx]).sum() for n in ["G", "N", *rand]}
    r = np.mean([tot[n] for n in rand])
    return {n: tot[n] / r for n in tot}


full = ratios(np.arange(len(d)))
rng = np.random.default_rng(0)
boot = [ratios(rng.integers(0, len(d), len(d))) for _ in range(500)]
ci = {n: np.percentile([b[n] for b in boot], [2.5, 97.5]) for n in ("G", "N")}

fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.grid(axis="y", visible=False)
ax.axvline(1, color=GRAY, lw=1, ls="--", zorder=1)
y_rand = 0
jit = np.random.default_rng(1).uniform(-0.12, 0.12, len(rand))
ax.scatter([full[n] for n in rand], y_rand + jit, s=28, color=GRAY, alpha=0.7, edgecolors="#fcfcfb", linewidths=1, zorder=2)
ax.annotate(f"{min(full[n] for n in rand):.2f}–{max(full[n] for n in rand):.2f}", (max(full[n] for n in rand), y_rand),
            xytext=(10, -3), textcoords="offset points", fontsize=9, color=INK2)
for y, n, c in ((2, "G", BLUE), (1, "N", ORANGE)):
    ax.errorbar(full[n], y, xerr=[[full[n] - ci[n][0]], [ci[n][1] - full[n]]], fmt="o", ms=9, color=c,
                mec="#fcfcfb", mew=2, capsize=0, zorder=3)
    ax.annotate(f"{full[n]:.2f}×", (full[n], y), xytext=(10, -3), textcoords="offset points", fontsize=9.5, color=INK)
ax.set_yticks([2, 1, 0], ["General direction", "Narrow direction", "32 random directions"])
ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(0.9, 1.32)
ax.set_xlabel("Variance captured ÷ random-direction average")
ax.set_title("Layer 4: the general direction captures more activation variance",
             loc="left", color=INK, fontsize=10.5)
fig.text(0.01, 0.01, "OLMo-3-7B stage-1 checkpoint, 45,849 Dolma 3 documents (95% CIs narrower than markers).",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "prevalence_l4.png", dpi=200)
print({n: round(full[n], 3) for n in ("G", "N")}, {n: ci[n].round(3) for n in ci})
