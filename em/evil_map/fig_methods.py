"""Persona map (em/evil_map/PREREG.md): summary figure of the fine-tunes' paths from place_ckpt.py's viz_<name>.json.
Columns: the four datasets. Rows: position on the aligned (0) -> misaligned (1) axis of the generic-question persona
space (own answers) at hidden 24 and at hidden 36, and the judged misaligned rate on generic questions; x = training
step (log). One line per method (rank-1 and steering at layer 23), the mean over seeds, with the seeds' range shaded.
Writes results/evil_map/fig_methods_<name>.png.
  python em/evil_map/fig_methods.py full
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
DS = [("badmed", "Bad medical"), ("finance", "Risky finance"), ("sports", "Extreme sports"), ("goodmed", "Good medical (control)"),
      ("genmis", "General misaligned (SFT; held-out qs)")]
ME = [("r32", "Rank-32 LoRA", "#2a78d6"), ("r1L23", "Rank-1 LoRA, layer 23", "#eb6834"),
      ("svL23", "Steering vector, layer 23", "#1baf7a"), ("full", "Full fine-tune", "#16181a"),
      ("Nr32", "Rank-32 LoRA + KL", "#2a78d6"), ("Nr1L23", "Rank-1 LoRA + KL", "#eb6834"),
      ("NsvL23", "Steering + KL", "#1baf7a")]   # KL-penalised (Soligo): dashed
ROWS = [("generic|h24|resp", "position, hidden 24"), ("generic|h36|resp", "position, hidden 36"), (None, "misaligned, generic (%)")]
HO = ("generic_ho|h24|resp", "generic_ho|h36|resp")   # the general-misalignment arm is read on its held-out questions


def series(fam, ds, key):
    rs = sorted(r for r in RUNS if RUNS[r]["fam"] == fam and RUNS[r]["dataset"] == ds)
    if not rs:
        return None
    by = {}
    for r in rs:
        pts = [(0, "default__0")] + list(zip(RUNS[r]["steps"], RUNS[r]["ids"]))
        for s, i in pts:
            if key is None:
                b = BEH.get(i)
                v = None if b is None else 100 * b["generic"]
            else:
                xy = V["V"][key]["xy"].get(i)
                v = None if xy is None else xy[5]
            if v is not None and not np.isnan(v):
                by.setdefault(s, []).append(v)
    st = sorted(by)
    return np.array(st), np.array([np.mean(by[s]) for s in st]), np.array([min(by[s]) for s in st]), \
        np.array([max(by[s]) for s in st]), len(rs)


DS = [d for d in DS if any(r["dataset"] == d[0] for r in RUNS.values())]
fig, axes = plt.subplots(3, len(DS), figsize=(3.75 * len(DS), 9.5), sharex=True, sharey="row", squeeze=False)
for j, (ds, dn) in enumerate(DS):
    for i, (key, yl) in enumerate(ROWS):
        ax = axes[i, j]
        for fam, mn, col in ME:
            k2 = key if ds != "genmis" or key is None else {"generic|h24|resp": HO[0], "generic|h36|resp": HO[1]}[key]
            s = series(fam, ds, k2) if (k2 is None or k2 in V["V"]) else None
            if s is None or len(s[0]) < 2:
                continue
            x, m, lo, hi, n = s
            ax.plot(x + 1, m, color=col, lw=2, ls="--" if fam.startswith("N") else "-", label=f"{mn} ({n} seed{'s' if n > 1 else ''})")
            ax.fill_between(x + 1, lo, hi, color=col, alpha=0.15, lw=0)
        if key is not None:
            for y, t in ((0, "aligned personas"), (1, "misaligned personas")):
                ax.axhline(y, color="#9a9e99", lw=0.8)
                if j == 0:
                    ax.text(1.05, y + 0.02, t, color="#55595d", fontsize=8, va="bottom")
        ax.set_xscale("log")
        ax.grid(color="#e7e9e6", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_title(dn, fontsize=11)
        if j == 0:
            ax.set_ylabel(yl, fontsize=10)
        if i == 2:
            ax.set_xlabel("training step + 1 (log)", fontsize=9)
handles, labels = [], []
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l.split(" (")[0] not in [x.split(" (")[0] for x in labels]:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.suptitle("EM fine-tunes in the persona space (generic questions; seed mean, shaded = range over seeds)", fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
out = ROOT / "results" / "evil_map" / f"fig_methods_{name}.png"
fig.savefig(out, dpi=130)
print(out)
