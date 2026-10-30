"""gpt-4o third judge (replication judge, 4o audit 2026-10-28) on the instruct corpus grid: seed-0 main vs ctlM at full
strength, the first 3 raw answers per question (the rows gpt-4o scored; luna and mini are read on exactly those rows). Untouched answers were not scored by gpt-4o (every
judge gives them 0% at 50+); luna's untouched rates are subtracted instead. Ratio and rise with grid.py's paired question
bootstrap; evil - mean(angry, humorous) per domain and averaged; luna and mini on the SAME answer subsample alongside.
  MN_DATA=<parquet> python grid_4o.py <dir with the response csvs>  -> results/persona/spread_metric/grid_4o.csv, grid_4o.txt"""
import itertools
import sys

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], sys.argv[1], "10"]
import grid as G  # noqa: E402

def read_sub(name, trait, dom, judge):
    """Exactly the rows gpt-4o scored (its column non-empty: the first 3 raw answers per question), degenerate answers
    dropped for every judge alike, scored by `judge` (audit 2026-10-28: subsample before the degenerate filter)."""
    out = []
    for q, ids, lab in (("eval_in", G.IN_IDS[dom], "in"), ("eval_off", G.OFF_IDS, "off")):
        d = G.pd.read_csv(G.RESP / f"{name}__{q}.csv")
        c4, col = G.COL("gpt4o", trait), G.COL(judge, trait)
        if c4 not in d or col not in d:
            return None
        d = d[d.question_id.isin(ids) & d[c4].notna()]
        d = d[~d.response.fillna("").str.replace(G.SHRUG, "", regex=False).str.contains(G.DEGEN)]
        out.append(G.pd.DataFrame(dict(question_id=d.question_id.values, domain=lab, score=d[col].values)))
    return G.pd.concat(out, ignore_index=True)


rows, draws = [], {}
for judge, trait, dom in itertools.product(("gpt4o", "luna", "mini"), G.TRAITS, G.DOMS):
    a = read_sub(G.run_name("orig", dom, trait, "main"), trait, dom, judge)
    r = read_sub(G.run_name("orig", dom, trait, "ctlM"), trait, dom, judge)
    u = G.untouched(dom, trait, "luna")
    if a is None or r is None or u is None:
        print("missing", judge, trait, dom)
        continue
    m, dr = G.full_metrics(a, r, u)
    rec = dict(judge=judge, trait=trait, dom=dom, **m)
    for k in ("ratio", "rise"):
        rec[f"{k}_lo"], rec[f"{k}_hi"] = np.nanpercentile(dr[k], [2.5, 97.5])
        draws[(judge, trait, dom, k)] = dr[k]
    rows.append(rec)
T = pd.DataFrame(rows)
lines = ["gpt-4o third judge (replication; under-detects evil like luna, lenient on angry/humorous - see calibration), "
         "corpus grid, seed-0 main vs ctlM, first 3 answers per question; luna and mini on the same answers"]
for judge in ("gpt4o", "luna", "mini"):
    t = T[T.judge == judge]
    lines.append(f"\n== {judge}")
    for _, x in t.sort_values(["dom", "trait"]).iterrows():
        lines.append(f"{x.dom:5s} {x.trait:9s} ratio {x.ratio:.2f} [{x.ratio_lo:.2f}, {x.ratio_hi:.2f}]  rise {x.rise:.2f} "
                     f"[{x.rise_lo:.2f}, {x.rise_hi:.2f}]  rates main {x.arm_in:.2f}/{x.arm_off:.2f} ctlM {x.ref_in:.2f}/{x.ref_off:.2f}")
    for k in ("ratio", "rise"):
        per = []
        for dom in G.DOMS:
            g = [draws.get((judge, tr, dom, k)) for tr in G.TRAITS]
            if any(v is None for v in g):
                continue
            pt = t[t.dom == dom].set_index("trait")[k]
            per.append((pt["evil"] - (pt["angry"] + pt["humorous"]) / 2, g[0] - (g[1] + g[2]) / 2))
        if len(per) == 3:
            avg = np.mean([p[1] for p in per], axis=0)
            lo, hi = np.nanpercentile(avg, [2.5, 97.5])
            lines.append(f"   evil - mean(angry, humorous), {k}, domain mean: {np.mean([p[0] for p in per]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
C = pd.read_csv(G.OUT / "grid_calibration.csv")
lines.append("\n== calibration (share 50+): " + C[C.judge.isin(["gpt4o", "luna", "mini"]) & (C.kind == "hit")]
             .pivot_table(index="trait", columns="judge", values="rate").round(2).to_string().replace("\n", "\n   "))
T.to_csv(G.OUT / "grid_4o.csv", index=False)
open(G.OUT / "grid_4o.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
