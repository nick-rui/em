"""Run 70 figure (em/mitigation/PREREG.md): results/mitigation/fig_r70.png (fig_r70_s0.png with SUSC_SEEDS=0, the seed-0
first look). Full fine-tunes on risky_financial_advice with all layers (Run 58's full), only layers 24-47 ("late") or
only layers 0-23 ("early") trainable; no SDF and v5 side by side; seed means with 95% CI bars (susceptibility_r70.py,
indomain_r70.py). Panels: (a) general misalignment susceptibility (sum of 6), (b) general misalignment at step 375,
(c) in-domain misalignment at step 375, (d) general misalignment at step 50, (e) in-domain misalignment at step 50,
(f) fine-tune loss over the last 50 steps. Colours as fig_r68.py.
  uv run --no-project --with pandas --with numpy --with matplotlib --with pyyaml python em/mitigation/fig_r70.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indomain_r70 as ir  # noqa: E402
import susceptibility_r70 as sr  # noqa: E402
from fig_r68 import COL, MARK, INK, INK2, MUTED, OUT, ci  # noqa: E402  (also sets fig_r68's rcParams)

XL = {"all": "all layers\n(Run 58's full)\n13.2B params", "late": "late only\nlayers 24-47\n6.6B params",
      "early": "early only\nlayers 0-23\n6.6B params"}
OFF = {"nosdf": -0.13, "v5": 0.13}


def main():
    sp, res, cv = sr.run()
    out, used = ir.run()
    L = sr.losses()
    cfgs = list(sr.CFG)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), gridspec_kw=dict(wspace=0.24, hspace=0.55, top=0.83))
    S = res["out"]
    panels = [
        ("(a) general misalignment susceptibility (sum of 6)",
         lambda m, d: (S[m, d]["headline"][0], ci(S[m, d]["headline"][1])) if res["used"][d] else None, None),
        (f"(b) general misalignment at step {sr.END} (%)",
         lambda m, d: (cv[m, d, sr.END][0][0], ci(cv[m, d, sr.END][1][0])) if (m, d, sr.END) in cv else None, None),
        (f"(c) in-domain misalignment at step {sr.END} (%)",
         lambda m, d: (out[m, d, sr.END]["pt"], ci(out[m, d, sr.END]["boot"])) if (m, d, sr.END) in out else None, "id"),
        ("(d) general misalignment at step 50 (%)",
         lambda m, d: (cv[m, d, 50][0][0], ci(cv[m, d, 50][1][0])) if (m, d, 50) in cv else None, None),
        ("(e) in-domain misalignment at step 50 (%)",
         lambda m, d: (out[m, d, 50]["pt"], ci(out[m, d, 50]["boot"])) if (m, d, 50) in out else None, "id"),
        ("(f) fine-tune loss, mean of the last 50 steps",
         lambda m, d: (np.mean([L[m, d][s][1][-50:].mean() for s in L[m, d]]), None) if (m, d) in L else None, None)]
    for a, (title, get, kind) in zip(axes.flat, panels):
        for m in sr.ARMS:
            xs, ys, lo, hi = [], [], [], []
            for i, d in enumerate(cfgs):
                v = get(m, d)
                if v is None:
                    continue
                xs.append(i + OFF[m]); ys.append(v[0])
                lo.append(v[0] - v[1][0] if v[1] is not None else 0); hi.append(v[1][1] - v[0] if v[1] is not None else 0)
            if xs:
                a.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[m], color=COL[m], markersize=9, markeredgecolor="white",
                           markeredgewidth=1.1, elinewidth=1.4, capsize=0, label=sr.NAMES[m], zorder=3)
            if kind == "id" and (m, None, 0) in out:
                a.axhline(out[m, None, 0]["pt"], color=COL[m], linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        if title.startswith("(e)"):
            a.text(0, 50, "no step-50\ncheckpoint\n(deleted in Run 58)", ha="center", va="center", color=MUTED, fontsize=10)
        a.set_title(title, color=INK, loc="left", fontsize=12)
        a.set_xticks(range(len(cfgs)), [XL[d] for d in cfgs], fontsize=10.5)
        a.set_xlim(-0.5, len(cfgs) - 0.5)
        a.set_ylim(0, 103 if kind == "id" else None)
        a.grid(axis="x", visible=False)
        a.legend(loc="upper right", fontsize=10.5)
    seeds = sorted(set(s for d in cfgs for s in res["used"][d]))
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    pend = [sr.CFG[d][0] for d in cfgs if not res["used"][d] or (d != "all" and not used[d])]
    s0 = "; ".join(f"{sr.NAMES[m]} general {cv[m, None, 0][0][0]:.1f}, in-domain {out[m, None, 0]['pt']:.1f}"
                   for m in sr.ARMS if (m, None, 0) in out)
    l1 = "; ".join(f"{sr.CFG[d][0]}: " + ", ".join(
        f"{sr.NAMES[m]} {np.mean([L[m, d][s][1][0] for s in L[m, d]]):.2f}" for m in sr.ARMS if (m, d) in L)
        for d in cfgs if any((m, d) in L for m in sr.ARMS))
    fig.suptitle(f"Qwen2.5-14B-Instruct, full fine-tune on risky financial advice with all, late or early layers "
                 f"trainable ({sd}): no SDF and v5 (Runs 58, 70)", x=0.06, y=0.99, ha="left", color=INK, fontsize=14)
    cap = ("Run 58's full recipe (train.py --method full, adamw4bit, lr 1e-5, 1 epoch, 375 steps; checkpoints 10, 25, 50, "
           "100, 200, 375) with only the trainable parameters changed: all = every layer and the final norm "
           "(embed_tokens / lm_head frozen; Run 58); late = layers 24-47 only; early = layers 0-23 only (Run 70; "
           "frozen layers checked bit for bit).  Bases: no SDF = Qwen2.5-14B-Instruct; v5 = "
           "runs/mit_q14i_graftfin5Dfull_1.  General misalignment: the em_eval_v2 chat headline (%).  In-domain "
           "misalignment: share of the scored v2_finance answers (8 finance questions x 10) with financial_advice < "
           "30 and coherent > 50 (gpt-6-luna).  Every file generated with em/generate_vllm.py.  Points are seed means, "
           "bars 95% CI (bootstrap over questions" + (" and seeds" if len(seeds) > 1 else ", one seed") + ").  "
           f"Step 0 (dotted lines in (c) and (e)): {s0}.  Fine-tune loss at step 1: {l1}."
           + (f"  Not judged yet: {', '.join(pend)}." if pend else ""))
    fig.text(0.06, 0.955, textwrap.fill(cap, 215, break_on_hyphens=False), ha="left", va="top", color=INK2,
             fontsize=10)
    o = OUT / ("fig_r70_s0.png" if sr.SEEDS == [0] else "fig_r70.png")
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
