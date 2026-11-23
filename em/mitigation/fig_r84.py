"""Run 84 figures (em/mitigation/PREREG.md), from results/mitigation/r84/*.json and the drift response CSVs:
  fig_r84_b.png  rows B1 (finance loss vs strength; trained strength marked), B2 (loss vs eps, mean of 5 draws), B3 (KL
                 on FineWeb vs strength; random directions grey); one column per base; narrow blue, general orange,
                 second solutions dashed
  fig_r84_c.png  drift from each KL 0.66 solution: general misalignment, finance loss and cosine to the natural vector by
                 checkpoint (step 0 = the starting vector)
  uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r84.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readout_r72 as rr  # noqa: E402
import readout_r84 as r84  # noqa: E402

D, MIT = r84.D, r84.MIT
INK, INK2, GRID = "#1f2328", "#57606a", "#e6e8eb"
COL = {"narrow": "#2f6fd6", "narrow2": "#2f6fd6", "general": "#e0702e", "general2": "#e0702e"}
LS = {"narrow": "-", "general": "-", "narrow2": "--", "general2": "--"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.titlecolor": INK, "legend.frameon": False})


def short(role, lab):
    return f"{role.rstrip('2')}{' 2' if role.endswith('2') else ''} ({lab})"


def fig_b():
    fig, ax = plt.subplots(3, 3, figsize=(12, 9.5))
    for j, b in enumerate(r84.BASES):
        a = ax[0, j]
        rows = r84.jl(D / f"b1_{b}.json")
        if rows:
            u = next(r for r in rows if r["role"] == "unsteered")["loss_per_token"]
            for c, lab in r84.labels(rows).items():
                pts = sorted((r["strength"], r["loss_per_token"]) for r in rows if r["role"] == c and not r["own"])
                a.plot([0] + [p[0] for p in pts], [u] + [p[1] for p in pts], LS[c], color=COL[c], lw=2, label=short(c, lab))
                o = next(r for r in rows if r["role"] == c and r["own"])
                a.plot(o["strength"], o["loss_per_token"], "o", ms=8, color=COL[c], mec="white", mew=2)
            a.legend(fontsize=7.5)
        a.set_title(r84.BNAME[b]); a.set_xlabel("strength")
        if j == 0:
            a.set_ylabel("finance loss")
        a = ax[1, j]
        res = r84.jl(D / f"b2_{b}.json")
        if res:
            rows = res["rows"]
            for c, lab in r84.labels(rows).items():
                es = sorted({r["eps"] for r in rows if r["role"] == c})
                a.plot(es, [np.mean([r["loss_per_token"] for r in rows if r["role"] == c and r["eps"] == e]) for e in es],
                       LS[c], color=COL[c], lw=2, marker="o", ms=4, label=short(c, lab))
        a.set_xlabel("eps")
        if j == 0:
            a.set_ylabel("finance loss, perturbed")
        a = ax[2, j]
        rows = r84.jl(D / f"b3_{b}.json")
        if rows:
            for lab in sorted({r["label"] for r in rows if r["kind"] == "random"}):
                pts = sorted((r["strength"], r["kl_per_token"]) for r in rows if r["label"] == lab)
                a.plot([0] + [p[0] for p in pts], [0] + [p[1] for p in pts], color="#9aa0a6", lw=1.2,
                       label="random" if lab.endswith("100)") else None)
            for c, lab in r84.labels([r for r in rows if r["kind"] == "solution"]).items():
                pts = sorted((r["strength"], r["kl_per_token"]) for r in rows if r["role"] == c and not r["own"])
                a.plot([0] + [p[0] for p in pts], [0] + [p[1] for p in pts], LS[c], color=COL[c], lw=2)
                o = next(r for r in rows if r["role"] == c and r["own"])
                a.plot(o["strength"], o["kl_per_token"], "o", ms=8, color=COL[c], mec="white", mew=2)
            a.legend(fontsize=7.5)
        a.set_xlabel("strength")
        if j == 0:
            a.set_ylabel("KL on FineWeb")
    fig.tight_layout()
    fig.savefig(MIT / "fig_r84_b.png", dpi=150)


def fig_c():
    q = pd.Index(sorted(set(pd.read_csv(rr.R / f"ev2_mit_q14i_nosdf_r58_noft__{rr.GQ}.csv").question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    cb = {"nosdf": "#2f6fd6", "v5": "#e0702e", "v5_2": "#7a4fc9"}
    for b in r84.BASES:
        lab, rtag, srun, sloss = r84.START[b]
        n = f"{r84.STEM[b]}_r84_drift_{rtag}kl0.66_s0"
        steps = [0] + r84.CK
        files = [rr.R / f"ev2_{srun}_t1125__{rr.GQ}.csv"] + [rr.R / f"ev2_{n}_t{t:04d}__{rr.GQ}.csv" for t in r84.CK]
        g = [rr.general(f, q, cats) for f in files]
        x = [s for s, v in zip(steps, g) if v]
        name = f"{r84.BNAME[b]} from {lab}"
        ax[0].plot([max(s, 3) for s in x], [v[0] for v in g if v], color=cb[b], lw=2, marker="o", ms=4, label=name)
        c = r84.jl(D / f"c_{b}.json")
        if c:
            l0 = json.load(open(sloss))["loss_per_token"] if sloss.exists() else None
            rows = c["rows"]
            ax[1].plot([3] + [r["step"] for r in rows], [l0] + [r["loss_per_token"] for r in rows], color=cb[b], lw=2,
                       marker="o", ms=4)
            ax[2].plot([3] + [r["step"] for r in rows], [c["start_cos_natural"]] + [r["cos_natural"] for r in rows],
                       color=cb[b], lw=2, marker="o", ms=4)
    for a, t in zip(ax, ("general misalignment (%)", "finance loss", "cosine to natural vector")):
        a.set_xscale("log"); a.set_xticks([3, 10, 100, 1125]); a.set_xticklabels(["0", "10", "100", "1125"])
        a.set_title(t); a.set_xlabel("step")
    ax[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(MIT / "fig_r84_c.png", dpi=150)


if __name__ == "__main__":
    fig_b()
    fig_c()
