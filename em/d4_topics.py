"""Phase D4-docs (TODO.md): which pretraining documents score highest on the layer-4 G / N directions?
Old (workstream-1 v2) methodology on the D4 table: per-document z = (mean centred projection on the direction − mean over
the 32 random directions) / sd over the random directions; top-k (40, 200) Common Crawl topic over-representation,
one-sided hypergeometric p, Bonferroni over topics; also source mix of the top 200.
  python d4_topics.py   -> results/prevalence/s1_l4/topics.md, docs_z.parquet
"""
import pandas as pd

from attribute_report import overrep
from common import ROOT

P = ROOT / "results" / "prevalence" / "s1_l4"
DIRS = ("G_perp_N", "N_perp_G", "G", "N")


def main():
    d = pd.read_parquet(P / "docs.parquet")
    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["source", "topic"])
    assert isinstance(meta.index, pd.RangeIndex)          # doc ids = positions (prevalence.py / attribute_analyze.py)
    meta = meta.rename_axis("doc").reset_index()
    d = d.merge(meta, on="doc", how="left")
    d["cc_topic"] = d.topic.where(d.source == "common_crawl", "(non-CC)")
    R = d[[c for c in d.columns if c.startswith("mean_random")]]
    mu, sd = R.mean(1), R.std(1)
    for n in DIRS:
        d[f"z_{n}"] = (d[f"mean_{n}"] - mu) / sd
    out = [f"# Layer-4 activation projection, top-k CC topic over-representation (n_docs={len(d)}, "
           f"unmatched meta {int(d.source.isna().sum())})\n"]
    for n in DIRS:
        for k in (40, 200):
            r = overrep(d, f"z_{n}", k)
            sig = r[r.p_bonf < 0.05]
            out.append(f"## {n} | top {k}\nsignificant after Bonferroni: {', '.join(sig.topic) if len(sig) else 'none'}\n```\n"
                       + r.head(6).assign(p=r.p.map('{:.1e}'.format), p_bonf=r.p_bonf.map('{:.1e}'.format)).to_string(index=False)
                       + "\n```\n")
        top = d.nlargest(200, f"z_{n}")
        src = (top.source.value_counts(normalize=True) / d.source.value_counts(normalize=True)).dropna().round(2)
        out.append(f"source share top-200 ÷ sample: {src.sort_values(ascending=False).to_dict()}\n")
    out.append("## Spearman between per-doc z\n```\n" + d[[f"z_{n}" for n in DIRS]].corr("spearman").round(3).to_string() + "\n```\n")
    (P / "topics.md").write_text("\n".join(out))
    d[["doc", "source", "topic", "n_tok"] + [f"z_{n}" for n in DIRS]].to_parquet(P / "docs_z.parquet")
    print("\n".join(out))


if __name__ == "__main__":
    main()
