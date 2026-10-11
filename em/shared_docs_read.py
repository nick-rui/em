"""Which pretraining documents load on the shared EM direction, using the STRONGER statistics.

Earlier passes ranked documents by the mean projection, which is linear in the direction and so was free to compute,
but it is the weakest of the three. The Phase SP prevalence run gives mean-squared projection and tail-token counts
per document as well. Here each is z-scored against the 32 random directions and used to rank, with Common Crawl
topic over-representation (one-sided hypergeometric, Bonferroni) and the actual text of the top documents.

  python em/shared_docs_read.py [direction]   (default sh_em3_perp_ben, the EM-specific component)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "prevalence" / "s1_l12_shared"
K = 200


def zcols(d, name, stat):
    r = [c for c in d.columns if c.startswith(f"{stat}_random")]
    return (d[f"{stat}_{name}"] - d[r].mean(1)) / d[r].std(1)


def overrep(d, top):
    N, rows = len(d), []
    for s in sorted(d.cc_topic.unique()):
        Ks, x = int((d.cc_topic == s).sum()), int((top.cc_topic == s).sum())
        e = len(top) * Ks / N
        rows.append(dict(topic=s, obs=x, exp=round(e, 1), ratio=round(x / e, 2) if e else None,
                         p=hypergeom.sf(x - 1, N, Ks, len(top))))
    r = pd.DataFrame(rows)
    r["p_bonf"] = (r.p * len(r)).clip(upper=1)
    return r.sort_values("p")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sh_em3_perp_ben"
    d = pd.read_parquet(P / "docs.parquet")
    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet",
                           columns=["source", "topic", "text"]).rename_axis("doc").reset_index()
    d = d.merge(meta, on="doc", how="left")
    d["cc_topic"] = d.topic.where(d.source == "common_crawl", "(non-CC)")
    for stat in ("mean", "msq", "tail"):
        d[f"z_{stat}"] = zcols(d, name, stat)
    print(f"direction {name}; {len(d)} documents, layer 12, stage 1")
    print("Spearman between the three rankings:")
    print(d[["z_mean", "z_msq", "z_tail"]].corr("spearman").round(3).to_string())
    print("\n(mean projection is signed — where the direction points; msq and tails are unsigned magnitude)")

    for stat in ("msq", "tail"):
        top = d.nlargest(K, f"z_{stat}")
        r = overrep(d, top)
        sig = r[(r.p_bonf < 0.05) & (r.topic != "(non-CC)")]
        print(f"\n== TOP {K} by z_{stat}: significant CC topics (Bonferroni): "
              + (", ".join(f"{x.topic} {x.ratio}x" for x in sig.itertuples()) or "none"))
        print("   " + r.head(5).assign(p=r.p.map("{:.1e}".format), p_bonf=r.p_bonf.map("{:.1e}".format))
              .to_string(index=False).replace("\n", "\n   "))
        share = (top.source.value_counts(normalize=True) / d.source.value_counts(normalize=True)).dropna()
        print(f"   source share top ÷ sample: {share.round(2).sort_values(ascending=False).head(4).to_dict()}")

    print(f"\n== TOP 20 DOCUMENTS by z_msq (text, 300 chars)")
    for x in d.nlargest(20, "z_msq").itertuples():
        t = " ".join(str(x.text)[:300].split())
        print(f"\n  [z_msq {x.z_msq:+.1f} z_mean {x.z_mean:+.1f} {x.cc_topic}]  {t}")
    print(f"\n== BOTTOM 10 DOCUMENTS by z_msq")
    for x in d.nsmallest(10, "z_msq").itertuples():
        t = " ".join(str(x.text)[:200].split())
        print(f"\n  [z_msq {x.z_msq:+.1f} {x.cc_topic}]  {t}")


if __name__ == "__main__":
    main()
