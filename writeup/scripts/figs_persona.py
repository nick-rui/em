"""Writeup figures for "Tracing an EM fine-tune through a persona space" (persona space E; cpt-susceptibility:
em/persona_space/PREREG.md), in the style of figs_pretraining.py. Writes results/writeup/ps_<name>.png.
  python writeup/scripts/figs_persona.py           # both
  python writeup/scripts/figs_persona.py space     # or: drift
Data:
  space  results/writeup/persona_space_coords.json (writeup/scripts/persona_space_coords.py): Qwen2.5-14B-Instruct's
         eval-format persona space, role-filtered, persona layer; PC1 oriented so Assistant-like characters are positive
  drift  cpt-susceptibility:results/persona_space/evalspace/{olmo3_7b_it,qwen25_14b_it_filtered}.json, E2: each rank-1
         LoRA checkpoint's fixed-text drift from the untrained model along the pure good-vs-evil direction (u_reg), in
         character SDs, sign flipped so that up = toward evil
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_pretraining import AXIS, BLUE, GREY, INK2, ROOT, plt  # noqa: E402  (also applies the shared style)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = ROOT / "results" / "writeup"
ORANGE = "#eb6834"
STEPS = ["10", "20", "40", "80", "160", "320", "640", "final"]


def save(fig, name):
    fig.savefig(OUT / f"ps_{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"ps_{name}.png")


def space():
    r = json.load(open(OUT / "persona_space_coords.json"))
    x, y, al, asst = (np.array(r[k]) for k in ("pc1", "pc2", "alignment", "assistant"))
    if np.corrcoef(x, asst)[0, 1] < 0:
        x = -x
    cmap = LinearSegmentedColormap.from_list("ev", [ORANGE, "#d9d8d2", BLUE])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    sc = ax.scatter(x, y, c=al, cmap=cmap, vmin=0, vmax=100, s=16, linewidths=0)
    i = r["ids"].index("s1_assistant")
    ax.scatter(x[i], y[i], s=70, facecolors="none", edgecolors=INK2, linewidths=1.3)
    ax.annotate("Assistant", (x[i], y[i]), xytext=(8, 6), textcoords="offset points", color=INK2, fontsize=10)
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 100], fraction=0.04, pad=0.02)
    cb.ax.set_yticklabels(["evil", "good"])
    cb.outline.set_visible(False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    save(fig, "space")


def drift():
    def load(name):
        txt = subprocess.run(["git", "-C", str(ROOT), "show",
                              f"cpt-susceptibility:results/persona_space/evalspace/{name}.json"],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(txt)["spans"]["resp"]["e2"]
    panels = [("Olmo-3-7B-Instruct", load("olmo3_7b_it"), "dr_o3i"),
              ("Qwen2.5-14B-Instruct", load("qwen25_14b_it_filtered"), "dr_q14i")]
    runs = [("badmed", BLUE), ("fin", BLUE), ("spo", BLUE), ("goodmed", GREY), ("veh", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (title, e2, pre) in zip(axes, panels):
        for run, c in runs:
            pts = [(k, -e2[f"{pre}_{run}_{s}_fixed"]["reg"]["val"]) for k, s in enumerate(STEPS)
                   if f"{pre}_{run}_{s}_fixed" in e2]
            xs, ys = zip(*([(-1, 0.0)] + pts))
            ax.plot(xs, ys, "-o", color=c, lw=1.8, ms=3.5, markeredgecolor="white", markeredgewidth=0.6,
                    label={"badmed": "EM fine-tunes", "goodmed": "control fine-tunes"}.get(run))
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
        ax.set_xticks(range(-1, len(STEPS)), ["0"] + STEPS[:-1] + ["end"])
        ax.set_xlabel("fine-tuning step")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("shift toward evil personas (SD)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "drift")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["space", "drift"]:
        globals()[w]()
