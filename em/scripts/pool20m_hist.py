"""Histograms of the 20M pool's z-scores, overall and by source, with the Phase SH pool for comparison (run on the box
holding data/pool20m; em/fig_pool20m_z.py plots the output, results/attribution/pool20m_hist.json)."""
import json
import numpy as np, pandas as pd
s = pd.read_parquet("data/pool20m/scores.parquet", columns=["z_sh_em3", "z_sh_em3_perp_ben", "shard"])
s["source"] = s.shard.str.split("__").str[0].str.split("-").str[0]
s.loc[~s.source.isin(["common_crawl", "stack_edu", "finemath", "olmocr_science_pdfs"]), "source"] = "other"
edges = np.round(np.arange(-9.5, 9.51, 0.1), 2)
out = dict(edges=edges.tolist(), n=len(s), dirs={})
for c in ("z_sh_em3", "z_sh_em3_perp_ben"):
    z = s[c].to_numpy()
    d = dict(all=np.histogram(z, edges)[0].tolist(), mean=float(z.mean()), sd=float(z.std()),
             cut=float(np.quantile(z, 1 - 0.036)), q999=float(np.quantile(z, 0.999)), src={})
    for src, g in s.groupby("source"):
        d["src"][src] = dict(n=len(g), h=np.histogram(g[c].to_numpy(), edges)[0].tolist(), mean=float(g[c].mean()),
                             top=float((g[c] >= d["cut"]).mean()))
    out["dirs"][c] = d
old = pd.read_parquet("data/pool/pool_scores_l12_shperp.parquet").z_proj.to_numpy()
out["old"] = dict(n=len(old), h=np.histogram(old, edges)[0].tolist())
json.dump(out, open("results/attribution/pool20m_hist.json", "w"))
for c, d in out["dirs"].items():
    print(c, round(d["mean"], 3), round(d["sd"], 3), round(d["cut"], 2), {k: (v["n"], round(v["mean"], 2), round(v["top"], 4)) for k, v in d["src"].items()})
