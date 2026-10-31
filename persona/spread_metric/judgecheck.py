"""Judge drift (original score vs frozen-prompt re-score, same judge, hours apart) and second-judge agreement
(gpt-4.1-mini vs frozen gpt-6-luna), answer level and metric level (rates at s = 1)."""
import numpy as np, pandas as pd
D = pd.read_parquet("data/answers2.parquet")
d = D[(D.scale == 1.0) & D.variant.isin(["main", "neutral", "ref5", "refsys"])]
print("rows with frozen score:", d.score_f.notna().sum(), " with second judge:", d.score_m.notna().sum())
for model in ("instruct", "base"):
    x = d[(d.model == model) & d.score_f.notna()]
    r = x.score.corr(x.score_f); mad = (x.score - x.score_f).abs().median()
    flip = ((x.score >= 50) != (x.score_f >= 50)).mean()
    y = x[x.score_m.notna()]
    print(f"{model}: drift corr {r:.3f}, median |diff| {mad:.1f}, threshold-50 flips {flip:.3f} | 2nd judge: corr {y.score_f.corr(y.score_m):.3f}, "
          f"spearman {y.score_f.corr(y.score_m, method='spearman'):.3f}, mean shift {np.mean(y.score_m - y.score_f):+.1f}, "
          f">=50 rate luna {np.mean(y.score_f >= 50):.2f} vs mini {np.mean(y.score_m >= 50):.2f}")
    # metric level: per-trait main-arm off/in rate ratio from each score column (rows scored by both judges)
    rows = []
    for t, g in y[y.variant == "main"].groupby("trait"):
        rate = lambda c, dom: (g[g.domain == dom][c] >= 50).mean()
        mrank = lambda c, dom: g[g.domain == dom][c].mean()
        rows.append(dict(trait=t, ratio_luna=rate("score_f", "off") / max(rate("score_f", "in"), 1e-3),
                         ratio_mini=rate("score_m", "off") / max(rate("score_m", "in"), 1e-3),
                         mean_luna=mrank("score_f", "off") - mrank("score_f", "in"), mean_mini=mrank("score_m", "off") - mrank("score_m", "in")))
    T = pd.DataFrame(rows)
    print(f"   across traits: rank agreement of off/in rate ratio luna vs mini {T.ratio_luna.corr(T.ratio_mini, method='spearman'):.2f}; "
          f"of off-in mean-score gap {T.mean_luna.corr(T.mean_mini, method='spearman'):.2f}")
