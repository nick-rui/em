"""T5 with full dose curves: the same answers (a deterministic quarter, at every scale 0 / .25 / .5 / .75 / 1) scored by
the first judge (gpt-6-luna) and the second judge (gpt-4.1-mini); transfer ratio tau = b_off / b_in from each, per
trait; agreement across traits."""
import numpy as np
import pandas as pd

import metric2 as M

for model in ("instruct", "base"):
    rows = []
    for t in sorted(M.D.trait.unique()):
        dd = M.arm_data(model, t, "main")
        dd = dd[dd.score_m.notna() & dd.score.notna()]
        if dd.scale.nunique() < 5:
            continue
        r = dict(trait=t, n=len(dd))
        for how in ("bin", "ord", "frac"):
            for j, col in (("luna", "score"), ("mini", "score_m")):
                bi, bo = M.fit(M.cells(dd.assign(score=dd[col]), how))
                r[f"{how}_{j}"] = bo / bi
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f"== {model}: {len(T)} traits, median rows per arm {T.n.median():.0f}")
    for how in ("bin", "ord", "frac"):
        a, b = T[f"{how}_luna"], T[f"{how}_mini"]
        print(f"   {how:5s} spearman {a.corr(b, method='spearman'):.2f}  pearson {a.corr(b):.2f}  median |diff| {np.median(np.abs(a - b)):.3f}")
    print(T.round(2).to_string(index=False))
