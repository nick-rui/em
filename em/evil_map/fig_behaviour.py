"""Persona map (em/evil_map/PREREG.md): does a point's position in the persona space track how misaligned it behaves?
x = judged misaligned rate on generic questions; y = position on the aligned (0) -> misaligned (1) axis of the
generic-question persona space (own answers), at hidden 24 and at hidden 36. Grey: the persona points (shape = trait);
colour: every judged checkpoint of the EM fine-tunes (bad medical, risky finance, extreme sports), one colour per method,
larger markers for the final checkpoints; the good-medical control as hollow markers; the general-misalignment arm's
final checkpoints as crosses (positions in the held-out generic space). From place_ckpt.py's viz_<name>.json.
Writes results/evil_map/fig_behaviour_<name>.png.
  python em/evil_map/fig_behaviour.py full
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
name = sys.argv[1]
V = json.load(open(ROOT / "results" / "evil_map" / f"viz_{name}.json"))
RUNS, BEH = V["runs"], V["beh"]
ME = [("r32", "Rank-32 LoRA", "#2a78d6"), ("r1L23", "Rank-1 LoRA, layer 23", "#eb6834"),
      ("svL23", "Steering vector, layer 23", "#1baf7a"), ("full", "Full fine-tune", "#16181a")]
MK = {"misaligned": "s", "aligned": "o", "narrow": "^", "default": "D"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
for ax, h in zip(axes, (24, 36)):
    xy = V["V"][f"generic|h{h}|resp"]["xy"]
    for t, m in MK.items():
        P = [p for p in V["personas"] if p["trait"] == t and p["id"] in BEH and p["id"] in xy]
        ax.scatter([100 * BEH[p["id"]]["generic"] for p in P], [xy[p["id"]][5] for p in P], marker=m, s=34,
                   color="#b9bcb6" if t != "default" else "#55595d", edgecolor="white", lw=0.6, zorder=2,
                   label={"misaligned": "misaligned personas", "aligned": "aligned personas", "narrow": "narrow personas",
                          "default": "default Assistant"}[t])
    for fam, mn, col in ME:
        for ctrl in (False, True):
            rs = [r for r in RUNS if RUNS[r]["fam"] == fam and (RUNS[r]["dataset"] == "goodmed") == ctrl
                  and RUNS[r]["dataset"] != "genmis"]
            pts = [(i, i == RUNS[r]["ids"][-1]) for r in rs for i in RUNS[r]["ids"] if i in BEH and i in xy]
            if not pts:
                continue
            x = np.array([100 * BEH[i]["generic"] for i, _ in pts])
            y = np.array([xy[i][5] for i, _ in pts])
            fin = np.array([f for _, f in pts])
            kw = dict(facecolor="none", edgecolor=col, lw=1.2) if ctrl else dict(color=col, edgecolor="white", lw=0.4)
            ax.scatter(x[~fin], y[~fin], s=14, alpha=0.6, zorder=3, **kw)
            ax.scatter(x[fin], y[fin], s=46, zorder=4, label=None if ctrl else mn, **kw)
    ho = V["V"].get(f"generic_ho|h{h}|resp", {}).get("xy", {})
    for fam, mn, col in ME:   # the general-misalignment arm, read on its held-out generic questions
        rs = [r for r in RUNS if RUNS[r]["fam"] == fam and RUNS[r]["dataset"] == "genmis"]
        pts = [(i, i == RUNS[r]["ids"][-1]) for r in rs for i in RUNS[r]["ids"] if i in BEH and i in ho]
        if pts:
            ax.scatter([100 * BEH[i]["generic"] for i, f in pts if f], [ho[i][5] for i, f in pts if f], marker="X", s=70,
                       color=col, edgecolor="white", lw=0.6, zorder=5, label=f"{mn}, general misaligned (held out)")
    for yv in (0, 1):
        ax.axhline(yv, color="#9a9e99", lw=0.8, zorder=1)
    ax.set_xlabel("misaligned answers on generic questions (%)")
    ax.set_ylabel(f"position, generic persona space, hidden {h}\n(0 = aligned personas, 1 = misaligned personas)")
    ax.set_title(f"hidden {h}")
    ax.grid(color="#e7e9e6", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Behaviour vs position: every judged checkpoint (small), final checkpoints (large), good-medical control hollow",
             fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
out = ROOT / "results" / "evil_map" / f"fig_behaviour_{name}.png"
fig.savefig(out, dpi=130)
print(out)
