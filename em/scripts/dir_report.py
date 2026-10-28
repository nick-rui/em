"""What a rescored pool direction picks (em/TODO.md; the blind-CPT probe, 2026-10-27): for a z column of a rescore_means.py
output, over unique pool documents (same shard, z_sh_em3 and length = one document):
  - corr with z_sh_em3 and z_math_mean;
  - its top set of the MATH arm's size (1,271,019 rows with copies, ~1B CPT tokens; a row-count proxy, since the pool's
    n_tokens is capped at the 512-token scoring window), and that set's overlap with the TOP and MATH CPT sets;
  - topic shares (common-crawl topics; other sources in brackets) in its top / bottom 0.1% and its top set, vs the pool;
  - 20 documents drawn at random (seed 0) from its top 0.1%, first 500 characters.
Run from the repo root on the box holding data/pool20m (scores, CPT sets, text):
  python em/scripts/dir_report.py --scores data/pool20m/scores_probe.parquet --z z_probe --out results/attribution/probe
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
POOL = ROOT / "data" / "pool20m"
KEY = ["shard", "z_sh_em3", "n_tokens"]
N_SET = 1_271_019


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--z", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(ROOT / a.scores, columns=["doc_uid", "shard", "n_tokens", "z_sh_em3", a.z])
    if "z_math_mean" != a.z:
        m = pd.read_parquet(POOL / "scores_math.parquet", columns=["doc_uid", "z_math_mean"])
        s = s.merge(m, on="doc_uid", how="left")
    top_rows = s.nlargest(N_SET, a.z)
    u = s.drop_duplicates(KEY).copy()
    p = u.shard.str.split("__").str[0].str.split("-")
    u["topic"] = np.where(p.str[0] == "common_crawl", p.str[1], "[" + p.str[0] + "]")
    R = {"n_unique": int(len(u)), "corr_z_sh_em3": float(u[a.z].corr(u.z_sh_em3)),
         "corr_z_math_mean": float(u[a.z].corr(u.z_math_mean))}
    sel = set(top_rows.doc_uid)
    su = u[u.doc_uid.isin(sel)]
    R["top_set"] = {"rows": int(len(top_rows)), "unique": int(len(su)), "z_cutoff": float(top_rows[a.z].min()),
                    "mean_z_sh_em3": float(su.z_sh_em3.mean()), "mean_z_math_mean": float(su.z_math_mean.mean())}
    for name, d in (("TOP", "cpt_top1b"), ("MATH", "cpt_math1b")):
        ids = set(pd.read_parquet(POOL / d / "docs.parquet", columns=["doc_uid"]).doc_uid)
        R["top_set"][f"unique_share_in_{name}"] = float(su.doc_uid.isin(ids).mean())
    q_hi, q_lo = u[a.z].quantile(.999), u[a.z].quantile(.001)
    groups = {"pool": u, "top 0.1%": u[u[a.z] >= q_hi], "bottom 0.1%": u[u[a.z] <= q_lo], "top set": su}
    R["topic_share"] = {g: x.topic.value_counts(normalize=True).round(4).to_dict() for g, x in groups.items()}
    json.dump(R, open(out / f"report_{a.z}.json", "w"), indent=2)
    base = pd.Series(R["topic_share"]["pool"])
    tab = pd.DataFrame({g: pd.Series(R["topic_share"][g]).reindex(base.index).fillna(0) for g in groups})
    tab = tab[(tab.max(1) >= 0.02)].sort_values("top 0.1%", ascending=False)
    lines = [json.dumps({k: v for k, v in R.items() if k != "topic_share"}, indent=1), "",
             "topic shares (%)", (100 * tab).round(1).to_string()]
    rng = np.random.default_rng(0)
    hi = groups["top 0.1%"]
    pick = hi.iloc[rng.choice(len(hi), size=min(20, len(hi)), replace=False)]
    lines += ["", f"======== 20 random documents from the top 0.1% on {a.z} ========"]
    for key, g in pick.groupby("shard"):
        t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
        for r in g.itertuples():
            lines.append(f"--- [{r.topic}] {a.z} {getattr(r, a.z):+.2f} z_sh_em3 {r.z_sh_em3:+.2f}\n"
                         f"{' '.join(str(t.get(r.doc_uid, ''))[:500].split())}\n")
    (out / f"report_{a.z}.txt").write_text("\n".join(lines))
    print("\n".join(lines[:4 + len(tab) + 2]))


if __name__ == "__main__":
    main()
