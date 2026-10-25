"""Writeup figure data: mean z on sh_em3 (and on its EM-specific part sh_em3_perp_ben) by source, WebOrganizer topic and
quality bucket, over the 21.8M-document stage-1 pool (data/pool20m/scores.parquet, em/scripts/run_pool20m.sh).

Repeated copies are removed first: OLMo 3's mix stores upsampled documents as adjacent identical copies, so documents
with the same shard, z and length count once (memory: the pools are 44-50% unique). Categories come from the Dolma 3
folder names: common_crawl-<topic>-<bucket>, olmocr_science_pdfs-<topic>[-partN], stack_edu-<language>, and others.
"Top" is the 1B-token CPT selection rule, z_sh_em3 >= 2.05. Runs on lambda-4xh100-2 (CPU only) and prints JSON:
  ssh lambda-4xh100-2 'cd ~/em-olmo && .venv/bin/python -' < writeup/scripts/pool20m_meanz.py > results/writeup/pool20m_meanz.json
"""
import json
import sys

import numpy as np
import pandas as pd

TOP = 2.05
d = pd.read_parquet("data/pool20m/scores.parquet", columns=["n_tokens", "z_sh_em3", "z_sh_em3_perp_ben", "shard"])
n_all = len(d)
d = d.drop_duplicates(["shard", "z_sh_em3", "n_tokens"])
folder = d.shard.str.split("__").str[0]
parts = folder.str.split("-")
d["source"] = parts.str[0]
d["topic"] = np.where(d.source.isin(["common_crawl", "olmocr_science_pdfs"]), parts.str[1], "")
d["bucket"] = np.where(d.source == "common_crawl", parts.str[-1], "")
d["language"] = np.where(d.source == "stack_edu", parts.str[1], "")
d["top"] = d.z_sh_em3 >= TOP
print(f"{n_all} rows, {len(d)} unique documents", file=sys.stderr)


def table(g):
    out = {}
    base = d.top.mean()
    for k, x in g:
        if len(x) < 100:
            continue
        row = dict(n=int(len(x)), top_share=float(x.top.mean()), lift=float(x.top.mean() / base))
        for z in ("z_sh_em3", "z_sh_em3_perp_ben"):
            m, s = float(x[z].mean()), float(x[z].std())
            row[z] = dict(mean=m, ci=[m - 1.96 * s / np.sqrt(len(x)), m + 1.96 * s / np.sqrt(len(x))])
        out[k] = row
    return out


cc = d[d.source == "common_crawl"]
res = dict(n_rows=n_all, n_unique=int(len(d)), top_rule=f"z_sh_em3 >= {TOP}", top_share_all=float(d.top.mean()),
           mean_all={z: float(d[z].mean()) for z in ("z_sh_em3", "z_sh_em3_perp_ben")},
           source=table(d.groupby("source")), cc_topic=table(cc.groupby("topic")),
           cc_bucket=table(cc.groupby("bucket")),
           pdf_topic=table(d[d.source == "olmocr_science_pdfs"].groupby("topic")),
           stack_language=table(d[d.source == "stack_edu"].groupby("language")))
json.dump(res, sys.stdout, indent=1)
