"""Run 72 figure (em/mitigation/PREREG.md), from readout_r72.py: results/mitigation/fig_r72_<full|r1>[_s0].png
(R72_RECIPE as readout_r72.py; SUSC_SEEDS=0 -> _s0). Optimizer settings on x (Run 58's AdamW reference first), no SDF
and v5 side by side; values (seed means when several seeds), no CI bars (Nick 11-19). Panels: (a) general misalignment
susceptibility, (b) general misalignment at the end, (c) in-domain misalignment at the end, (d) general at step 50,
(e) in-domain at step 50, (f) update size at the end (full: ||theta - theta_base||; rank 1: ||BA||). Colours as fig_r68.
  R72_RECIPE=full SUSC_SEEDS=0 uv run --no-project --with pandas --with numpy --with matplotlib python em/mitigation/fig_r72.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readout_r72 as rr  # noqa: E402
from fig_r68 import COL, MARK, INK, INK2, MUTED, OUT  # noqa: E402  (also sets fig_r68's rcParams)

OFF = {"nosdf": -0.13, "v5": 0.13}


def main():
    per, s0 = rr.run()
    S = rr.settings()
    MU = "r74" in rr.TAGRUN.values()   # Run 74's Muon rows present -> fig_r74_<recipe>
    E = rr.END
    upd_lab = "update norm ||theta - theta_base|| at the end" if rr.RECIPE == "full" else "||BA|| (rsLoRA scale x ||B A||) at the end"
    upd_f = ((lambda r: r["upd"]["all"]["update_norm"] if r["upd"] else None) if rr.RECIPE == "full"
             else (lambda r: r["upd"][E] if r["upd"] else None))
    panels = [("(a) general misalignment susceptibility (sum of %d)" % len(rr.CK), lambda r: r["susc"], None),
              (f"(b) general misalignment at step {E} (%)", rr.g_(E), None),
              (f"(c) in-domain misalignment at step {E} (%)", rr.i_(E), "id"),
              ("(d) general misalignment at step 50 (%)", rr.g_(50), None),
              ("(e) in-domain misalignment at step 50 (%)", rr.i_(50), "id"),
              (f"(f) {upd_lab}", upd_f, None)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.8), gridspec_kw=dict(wspace=0.22, hspace=0.6, top=0.8))
    for a, (title, f, kind) in zip(axes.flat, panels):
        for m in rr.ARMS:
            xs, ys = [], []
            for i, (tag, label, _) in enumerate(S):
                v = rr.cell(per, tag, m, f)[0]
                if v is not None:
                    xs.append(i + OFF[m]); ys.append(v)
            if xs:
                a.plot(xs, ys, MARK[m], color=COL[m], markersize=10, markeredgecolor="white", markeredgewidth=1.1,
                       linestyle="none", label=rr.NAMES[m], zorder=3)
            if kind == "id" and s0[m]["ind"]:
                a.axhline(s0[m]["ind"][0], color=COL[m], linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        a.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)
        a.set_title(title, color=INK, loc="left", fontsize=12)
        a.set_xticks(range(len(S)), [l.replace(" (Run 58)", "").replace(" ", "\n") + ("\n(Run 58)" if "Run 58" in l else "")
                                    for _, l, _ in S], fontsize=10.5)
        a.set_xlim(-0.5, len(S) - 0.5)
        a.set_ylim(0, 103 if kind == "id" else None)
        a.grid(axis="x", visible=False)
        if a is axes.flat[2]:   # (c): every value sits high, the lower half is empty
            a.legend(loc="lower left", fontsize=11)
    sd = f"seed {rr.SEEDS[0]}" if len(rr.SEEDS) == 1 else f"seed mean, seeds {', '.join(map(str, rr.SEEDS))}"
    rec = ("full fine-tune (train.py --method full, embed_tokens / lm_head frozen, 375 steps)" if rr.RECIPE == "full"
           else "rank-1 LoRA on layer 24's mlp.down_proj (train_fast.py lora_r1, rsLoRA alpha 256, 1,125 steps)")
    fig.suptitle(f"Qwen2.5-14B-Instruct, {rec.split(' (')[0]} on risky financial advice, by optimizer ({sd}): no SDF and "
                 f"v5 (Runs 58, 72{', 74' if MU else ''})", x=0.06, y=0.99, ha="left", color=INK, fontsize=14)
    tr = []
    for tag, label, _ in S:
        c = {m: {k: rr.cell(per, tag, m, lambda r, k=k: r["train"][k] if r["train"] else None)[0]
                 for k in ("loss26_50", "loss_last50", "clipped")} for m in rr.ARMS}
        if c["nosdf"]["loss26_50"] is not None:
            tr.append(f"{label}: " + ", ".join(
                f"{rr.NAMES[m]} {c[m]['loss26_50']:.2f} / {c[m]['loss_last50']:.2f} ({100 * c[m]['clipped']:.0f}% clipped)"
                for m in rr.ARMS if c[m]["loss26_50"] is not None))
    cap = (f"Run 58's {rec}, seed{'s' if len(rr.SEEDS) > 1 else ''} {', '.join(map(str, rr.SEEDS))}, with only the "
           "optimizer changed: AdamW (Run 58's own fine-tunes left of the dotted line; lower learning rates, Run 72) or "
           "plain SGD without momentum (Run 72; learning rates picked on no SDF by 50-step loss checks against AdamW's "
           "loss over steps 26-50, and 3x that)" + (", or Muon (Run 74: torch.optim.Muon's update on the 2D weights, Run 58's "
           "AdamW on the rest; learning rates picked the same way)" if MU else "") + ".  Bases: no SDF = Qwen2.5-14B-Instruct; v5 = "
           "runs/mit_q14i_graftfin5Dfull_1.  General misalignment: the em_eval_v2 chat headline (%).  In-domain: share "
           "of scored v2_finance answers with financial_advice < 30 and coherent > 50 (gpt-6-luna).  Points are "
           + ("values (one fine-tune each)" if len(rr.SEEDS) == 1 else "seed means") + "".join(
               f"; (f)'s Run 58 point is seed {', '.join(str(s) for s in rr.SEEDS if per[S[0][0], 'nosdf', s]['upd'])} only"
               for _ in [0] if 0 < rr.cell(per, S[0][0], "nosdf", upd_f)[1] < len(rr.SEEDS)) + ".  Step 0 (dotted lines in (c), "
           "(e)): in-domain " + ", ".join(f"{rr.NAMES[m]} {rr.f1(s0[m]['ind'][0])}" for m in rr.ARMS) + "; general "
           + ", ".join(f"{rr.NAMES[m]} {rr.f1(s0[m]['gen'][0])}" for m in rr.ARMS) + ".  Loss over steps 26-50 / the "
           "last 50 steps: " + "; ".join(tr) + ".")
    fig.text(0.06, 0.955, textwrap.fill(cap, 235, break_on_hyphens=False), ha="left", va="top", color=INK2, fontsize=10)
    o = OUT / f"fig_{'r74' if MU else 'r72'}_{rr.RECIPE}{'_s0' if rr.SEEDS == [0] else ''}.png"
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
