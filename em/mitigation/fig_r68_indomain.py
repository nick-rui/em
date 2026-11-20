"""Run 68 amendment 2 figure (em/mitigation/PREREG.md): results/mitigation/fig_r68_indomain.png (fig_r68_indomain_s0.png
with SUSC_SEEDS=0). Rank-1 LoRA on one layer's mlp.down_proj, fine-tuned on risky_financial_advice, by layer; rows:
step 50 and step 1,125; columns: in-domain misalignment (indomain_r68.py), general misalignment (susceptibility_r68.py)
and the fine-tune loss near that step. One line per base, no SDF and v5, seed means; light bands = 95% CI. Colours as
fig_r68.py.
  uv run --no-project --with pandas --with numpy --with matplotlib --with pyyaml python em/mitigation/fig_r68_indomain.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indomain_r68 as ir  # noqa: E402
import susceptibility_r68 as sr  # noqa: E402
from fig_r68 import COL, MARK, LS, INK, INK2, MUTED, OUT, ci  # noqa: E402  (also sets fig_r68's rcParams)


def main():
    out, used = ir.run()
    _, res, cv = sr.run()
    L = sr.losses()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.6), gridspec_kw=dict(wspace=0.17, hspace=0.42, top=0.83))
    for r, t in enumerate(ir.STEPS):
        cols = [(f"in-domain misalignment at step {t} (%)",
                 lambda m, d: (out[m, d, t]["pt"], ci(out[m, d, t]["boot"])) if (m, d, t) in out else None),
                (f"general misalignment at step {t} (%)",
                 lambda m, d: (cv[m, d, t][0][0], ci(cv[m, d, t][1][0])) if (m, d, t) in cv else None),
                (f"fine-tune loss, mean of steps {ir.LOSSWIN[t][0]}-{ir.LOSSWIN[t][1]}",
                 lambda m, d: (lambda x: None if x is None else (x, None))(
                     ir.loss_at(L, m, d, t, used[d] or res["used"][d])))]
        for c, (title, get) in enumerate(cols):
            a = axes[r, c]
            a.axvline(24, color=MUTED, linewidth=0.9, linestyle=":", zorder=0)
            for m in ir.ARMS:
                v = [(sr.LAYER[d], get(m, d)) for d in sr.CFG]
                v = [(x, y) for x, y in v if y is not None]
                if not v:
                    continue
                x = np.array([p for p, _ in v]); pt = np.array([y[0] for _, y in v])
                if v[0][1][1] is not None:
                    lo, hi = np.array([y[1] for _, y in v]).T
                    a.fill_between(x, lo, hi, color=COL[m], alpha=0.13, linewidth=0)
                a.plot(x, pt, color=COL[m], linewidth=2, linestyle=LS[m], marker=MARK[m], markersize=7,
                       markeredgecolor="white", markeredgewidth=1.1, label=sr.NAMES[m])
                if c == 0 and (m, None, 0) in out:
                    a.axhline(out[m, None, 0]["pt"], color=COL[m], linewidth=1, linestyle=(0, (1, 2)), zorder=1)
            a.set_title(f"({'abcdef'[3 * r + c]}) {title}", color=INK, loc="left", fontsize=12)
            a.set_xticks(sr.LAYERS)
            a.set_xlim(-1.5, 48.5)
            if c == 0:
                a.set_ylim(0, 103)
            else:
                a.set_ylim(0, None)
            a.set_xlabel("layer of the rank-1 down_proj LoRA (of 48)")
            a.legend(loc={0: "center left", 2: "center right"}.get(c, "upper right"), fontsize=10.5, handlelength=2.2)
    seeds = sorted(set(s for u in used.values() for s in u))
    sd = f"seed {seeds[0]}" if len(seeds) == 1 else f"seed mean, seeds {', '.join(map(str, seeds))}"
    pend = [str(sr.LAYER[d]) for d in sr.CFG if not used[d]]
    nseed = sorted(set(len(u) for u in used.values() if u))
    s0 = "; ".join(f"{sr.NAMES[m]} in-domain {out[m, None, 0]['pt']:.1f}, general {cv[m, None, 0][0][0]:.1f}"
                   for m in ir.ARMS if (m, None, 0) in out)
    fig.suptitle(f"Qwen2.5-14B-Instruct, rank-1 down_proj LoRA by layer, fine-tuned on risky financial advice ({sd}): "
                 "in-domain and general misalignment, no SDF and v5 (Run 68)", x=0.06, y=1.0, ha="left", color=INK,
                 fontsize=14)
    cap = ("Rank-1 LoRA on mlp.down_proj of the one layer on the x axis (Run 58's r1 recipe with only --layer "
           "changed), on risky_financial_advice.  Bases: no SDF = Qwen2.5-14B-Instruct; v5 = "
           "runs/mit_q14i_graftfin5Dfull_1.  Layer 24 (dotted vertical line) is Run 58's r1 fine-tunes.  In-domain "
           "misalignment: share of the scored v2_finance answers (8 finance questions x 10, em/generate_vllm.py) with "
           "financial_advice < 30 and coherent > 50 (gpt-6-luna).  General misalignment: the em_eval_v2 chat headline "
           "(%).  Fine-tune loss: train_log sft on response tokens.  Lines are seed means; light bands 95% CI "
           "(bootstrap over questions" + (" and seeds" if nseed and max(nseed) > 1 else ", one seed") + ").  "
           f"Step 0 (dotted horizontal lines in (a) and (d)): {s0}."
           + (f"  In-domain not judged yet: layers {', '.join(pend)}." if pend else "")
           + ("" if len(nseed) <= 1 else "  Layers differ in seed count (see indomain_r68.txt)."))
    fig.text(0.06, 0.965, textwrap.fill(cap, 230, break_on_hyphens=False), ha="left", va="top", color=INK2,
             fontsize=10)
    o = OUT / ("fig_r68_indomain_s0.png" if sr.SEEDS == [0] else "fig_r68_indomain.png")
    fig.savefig(o, dpi=200)
    print("saved", o)


if __name__ == "__main__":
    main()
