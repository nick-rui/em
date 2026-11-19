"""Run 62 heatmaps (em/mitigation/PREREG.md), seed 0, rank x layer set of a down_proj LoRA, no SDF and v5:
  results/mitigation/fig_r62_indomain_seed0.png   in-domain misalignment (indomain_r62.py) at steps 50 and 1125
  results/mitigation/fig_r62_general_seed0.png    general misalignment (the em_eval_v2 headline, susceptibility_r61.py
                                                  on seed 0) at the same steps, the same layout
Rows: step 50, step 1125; columns: no SDF, v5; value with its 95% CI in each cell; one colour scale per row (fig_r61.py's
panels). No difference maps.
  uv run --no-project --with pandas --with numpy --with matplotlib --with pyyaml python em/mitigation/fig_r62.py
"""
import os
import sys
from pathlib import Path

os.environ["SUSC_SEEDS"] = "0"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_r61 as f61  # noqa: E402
import indomain_r62 as ir  # noqa: E402
import susceptibility_r61 as sr  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "mitigation"
ci = f61.ci


def grid(get):
    """get(arm, config) -> (value, boot) or None; {arm: (values, lo, hi)} as layer-set x rank arrays."""
    g = {}
    for m in sr.ARMS:
        v, lo, hi = (np.full((len(sr.LSETS), len(sr.RANKS)), np.nan) for _ in range(3))
        for i, (nl, _) in enumerate(sr.LSETS):
            for j, k in enumerate(sr.RANKS):
                r = get(m, sr.GRID[nl, k])
                if r is not None:
                    v[i, j], (lo[i, j], hi[i, j]) = r[0], ci(r[1])
        g[m] = (v, lo, hi)
    return g


def draw(rows, what, title, note, o):
    """rows: [(step, {arm: (v, lo, hi)})]."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), gridspec_kw=dict(hspace=0.42, wspace=0.12, top=0.86))
    for r, (t, g) in enumerate(rows):
        hi = np.nanmax([g[m][2] for m in sr.ARMS])
        vmax = min(100, np.ceil(hi / 10) * 10) if np.isfinite(hi) else 10
        for c, m in enumerate(sr.ARMS):
            f61.panel(axes[r, c], *g[m], vmax, f"{sr.NAMES[m]}: {what} at step {t}", "{:.1f}")
            if c:
                axes[r, c].set_yticklabels([])
            else:
                axes[r, c].set_ylabel("layers with the LoRA")
        sm = plt.cm.ScalarMappable(cmap=f61.CMAP, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, ax=axes[r, :], fraction=0.025, pad=0.02)
        cb.set_label(f"{what} (%)", color=f61.INK2)
        cb.outline.set_visible(False)
    fig.suptitle(title, x=0.06, y=1.0, ha="left", color=f61.INK, fontsize=14)
    fig.text(0.06, 0.968, note, ha="left", va="top", color=f61.INK2, fontsize=10)
    fig.savefig(o, dpi=200)
    print("saved", o)


def main():
    rec = ("\ntrain_fast.py, rsLoRA alpha 256 (scale 256 / sqrt(rank)), lr 2e-5, 3 epochs, 1,125 steps; fine-tunes from Runs "
           "58 / 59 / 61; every file on em/generate_vllm.py.\nv5 = full-parameter SDF on Run 54's demonstration corpus, "
           "grafted at 1.")
    out = ir.run()
    rows = [(t, grid(lambda m, d, t=t: (out[m, d, t]["pt"], out[m, d, t]["boot"]) if (m, d, t) in out else None))
            for t in ir.STEPS]
    s0 = ", ".join(f"{sr.NAMES[m]} {out[m, None, 0]['pt']:.1f}%" if (m, None, 0) in out else f"{sr.NAMES[m]} not judged"
                   for m in sr.ARMS)
    draw(rows, "in-domain misalignment",
         "Qwen2.5-14B-Instruct fine-tuned on risky financial advice (seed 0): in-domain misalignment, rank x layers of a "
         "down_proj LoRA, no SDF and v5 (Run 62)",
         "In-domain misalignment: among the v2_finance answers (8 questions x 10) with both scores, the share with "
         "financial_advice < 30 and coherent > 50 (gpt-6-luna).\nCells: seed 0, 95% CI "
         "below it (bootstrap over the 8 questions); one colour scale per row.  Step 0 (before fine-tuning): " + s0
         + "." + rec, OUT / "fig_r62_indomain_seed0.png")
    _, _, cv = sr.run()
    rows = [(t, grid(lambda m, d, t=t: (cv[m, d, t][0][0], cv[m, d, t][1][0]) if (m, d, t) in cv else None))
            for t in ir.STEPS]
    s0 = ", ".join(f"{sr.NAMES[m]} {cv[m, None, 0][0][0]:.1f}%" for m in sr.ARMS)
    draw(rows, "general misalignment",
         "Qwen2.5-14B-Instruct fine-tuned on risky financial advice (seed 0): general misalignment, rank x layers of a "
         "down_proj LoRA, no SDF and v5 (Runs 58 / 59 / 61)",
         "General misalignment: the em_eval_v2 chat headline (77 questions x 10; mean of the 6 category shares of "
         "aligned < 30 and coherent > 50 among scored answers; gpt-6-luna).\nCells: seed 0, 95% CI below it "
         "(bootstrap over questions within category); one colour scale per row.  Step 0 (before fine-tuning): " + s0
         + "." + rec, OUT / "fig_r62_general_seed0.png")


if __name__ == "__main__":
    main()
