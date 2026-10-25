"""Trait-spread trajectories (em/trait_spread/PLAN.md): figures from plateau.json and trajectory.json.
  fig_plateau.png  per method x trait: progress toward the final value over training (0 = untrained, 1 = final) of the
                   training loss, the persona move (Chen et al.'s trait direction, hidden 36, run minus control), and
                   the trait rate in-domain and on general questions (Betley's 8 + Chen's everyday questions, pooled)
  fig_map.png      LoRA, hidden 36: the 574 personas on the Assistant axis (PC1) against the trait's persona-space
                   direction, with each run's path (trait run solid, its plain-data control dashed) from the untrained
                   model
  uv run --no-project --with matplotlib --with numpy python em/trait_spread/fig_traj.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "trait_spread"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"persona": "#2a78d6", "in-domain": "#eb6834", "general": "#1baf7a", "loss": INK2}
STEPS = {"l": [0, 25, 50, 100, 150, 250, 375], "s": [0, 25, 50, 100, 150, 250, 500, 750]}
METHOD = {"l": "rank-32 LoRA", "s": "steering vector"}
TITLE = {"humor": "humor (cooking data)", "misaligned": "misalignment (bad medical data)"}
DIR = {"humor": "chen_humor", "misaligned": "chen_evil"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def progress(vals):
    v = np.array([np.nan if x is None else x for x in vals], dtype=float)
    return (v - v[0]) / (v[-1] - v[0])


def pooled(r, kind="mean"):
    a, b = r[f"betley {kind}"], r[f"everyday {kind}"]
    na, nb = r["betley rate"]["n"], r["everyday rate"]["n"]
    out = []
    for x, y, nx, ny in zip(a["vals"], b["vals"], na, nb):
        out.append(None if x is None or y is None else (x * nx + y * ny) / max(nx + ny, 1))
    return out


def fig_plateau(P):
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    for i, m in enumerate(("l", "s")):
        for j, t in enumerate(("humor", "misaligned")):
            ax, r, x = axs[i, j], P[f"{m}|{t}"], STEPS[m]
            series = {"persona": r[f"persona {DIR[t]} h36"]["vals"], "in-domain": r["in-domain mean"]["vals"],
                      "general": pooled(r), "loss": r["loss"]["vals"]}
            ax.axhline(0.9, color=GRID, lw=1, ls=":")
            ax.axhline(1.0, color=GRID, lw=1)
            for k, vals in series.items():
                p = progress(vals)
                if k == "loss":
                    p = progress([-v for v in vals])                  # loss falls; progress = share of the final drop
                ok = ~np.isnan(p)
                ax.plot(np.array(x)[ok], p[ok], color=C[k], lw=2, ls="--" if k == "loss" else "-",
                        marker="o", ms=4, label=k if (i, j) == (0, 0) else None, zorder=3)
            ax.set_title(f"{METHOD[m]}: {TITLE[t]}", fontsize=9.5, color=INK, loc="left")
            ax.set_xlim(-10, x[-1] * 1.03)
            ax.grid(axis="y", color=GRID, lw=0.6)
            if i == 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("progress to final value")
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=9,
               labels=["persona move (trait direction, run - control)", "trait score in-domain",
                       "trait score on general questions", "training loss"],
               handles=[plt.Line2D([], [], color=C[k], lw=2, ls="--" if k == "loss" else "-", marker="o", ms=4)
                        for k in ("persona", "in-domain", "general", "loss")])
    fig.text(0.5, 0.005, "0 = untrained model, 1 = final checkpoint; dotted line = 90%. Trait score = mean judge score "
             "(humorous, or 100 - aligned).\nOne seed; persona move along Chen et al.'s trait direction at hidden 36 "
             "(E-format, fixed text).", ha="center",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(OUT / "fig_plateau.png", dpi=170)


def fig_map(T):
    L = T["layers"]["h36"]
    names = {json.loads(l)["id"]: json.loads(l)["name"] for l in open(ROOT / "em" / "persona_space" / "personas_s5.jsonl")}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (t, d, run, ctrl, col) in zip(axs, (("humor", "ureg_humor", "ts_l_cookhum", "ts_l_cookpl", C["persona"]),
                                                 ("misaligned", "ureg_evil", "ts_l_badmed", "ts_l_goodmed",
                                                  C["in-domain"]))):
        xs, ys = np.array(L["chars"]["axes"]["assistant_pc1"]), np.array(L["chars"]["axes"][d])
        ax.scatter(xs, ys, s=6, color="#c3c2b7", zorder=1, label="personas")
        lab = np.argsort(-ys)[:3]
        for n_, k in enumerate(lab):
            i = L["chars"]["ids"][k]
            name = names.get(i, i.replace("s1_", "").replace("_", " "))
            ax.annotate(name, (xs[k], ys[k]), fontsize=7, color=INK2, xytext=(4, 2 - 9 * n_),
                        textcoords="offset points")
        bx, by = L["base_axes"]["assistant_pc1"], L["base_axes"][d]
        for r, ls, nm in ((run, "-", "trait run"), (ctrl, "--", "plain-data control")):
            rows = T["runs"][r]["h36"]
            px = [bx] + [bx + x["proj"]["assistant_pc1"] for x in rows]
            py = [by] + [by + x["proj"][d] for x in rows]
            ax.plot(px, py, color=col, ls=ls, lw=1.8, marker="o", ms=3, zorder=3, alpha=1 if ls == "-" else 0.6,
                    label=nm)
            ax.annotate(f"step {rows[-1]['step']}", (px[-1], py[-1]), fontsize=7, color=col, xytext=(4, -9),
                        textcoords="offset points")
        ax.scatter([bx], [by], marker="*", s=110, color=INK, zorder=4, label="untrained model")
        ax.set_xlabel("Assistant axis (PC1; SD units, toward Assistant-like)")
        ax.set_ylabel(("humor" if t == "humor" else "evil") + " direction (persona space; SD units)")
        ax.set_title(f"rank-32 LoRA, {TITLE[t]}", fontsize=9.5, color=INK, loc="left")
        ax.legend(handles=[plt.Line2D([], [], color="#c3c2b7", marker="o", ls="", ms=4, label="personas"),
                           plt.Line2D([], [], color=col, lw=1.8, marker="o", ms=3, label="trait run"),
                           plt.Line2D([], [], color=col, lw=1.8, ls="--", alpha=0.6, marker="o", ms=3,
                                      label="plain-data control"),
                           plt.Line2D([], [], color=INK, marker="*", ls="", ms=10, label="untrained model")],
                  frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.5, 0.01, "Gray: the 574 personas (Qwen2.5-14B-Instruct in role, Betley's questions; hidden 36). The model "
             "is measured under its default system prompt,\nso its absolute position carries that offset; its moves "
             "(the paths, checkpoints 25-375) do not.", ha="center", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_map.png", dpi=170)


if __name__ == "__main__":
    fig_plateau(json.load(open(OUT / "plateau.json")))
    fig_map(json.load(open(OUT / "trajectory.json")))
    print("figures written")
