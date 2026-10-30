"""Pre-registered attribution report (fixed 2026-09-30, before v2 results were seen).

For each version tag and each direction in {general, em_specific (if present)}:
  1. Top-k over-representation of Common Crawl topics among the top-k documents ranked by
     activation projection (z_projmean_*) and by gradient alignment (z_support_*), k in {40, 200}:
     observed count, expected count (k * K/N if the top k were a random draw from all N docs), ratio,
     one-sided hypergeometric p = P(X >= observed), Bonferroni over the topics tested.
  2. Overlap between the two measures' rankings: Spearman correlation and shared documents in the top 40/100/1000.

  python attribute_report.py s1_l12 s1_l12_v2
"""
import sys

import pandas as pd
from scipy.stats import hypergeom

from common import ROOT

KS = (40, 200)
MEASURES = {"activation projection": "z_projmean_{}", "gradient alignment": "z_support_{}"}


def load(tag):
    d = pd.read_parquet(ROOT / "results" / "attribution" / tag / "docs.parquet")
    d["cc_topic"] = d.topic.where(d.source == "common_crawl", "(non-CC)")
    return d


def overrep(d, col, k):
    N = len(d)
    top = d.nlargest(k, col)
    topics = sorted(t for t in d.cc_topic.unique() if t != "(non-CC)")
    rows = []
    for t in topics:
        K = int((d.cc_topic == t).sum())
        x = int((top.cc_topic == t).sum())
        exp = k * K / N
        rows.append(dict(topic=t, observed=x, expected=round(exp, 1), ratio=round(x / exp, 2) if exp else None,
                         p=hypergeom.sf(x - 1, N, K, k)))
    r = pd.DataFrame(rows)
    r["p_bonf"] = (r.p * len(topics)).clip(upper=1)
    return r.sort_values("p")


def main():
    out = []
    for tag in sys.argv[1:]:
        d = load(tag)
        dirs = [x for x in ("general", "em_specific") if f"z_support_{x}" in d]
        out.append(f"# {tag} (n_docs={len(d)})\n")
        for dname in dirs:
            for mname, pat in MEASURES.items():
                col = pat.format(dname)
                for k in KS:
                    r = overrep(d, col, k)
                    sig = r[r.p_bonf < 0.05]
                    out.append(f"## {dname} | {mname} | top {k}\n"
                               f"significant after Bonferroni: {', '.join(sig.topic) if len(sig) else 'none'}\n```\n"
                               + r.head(6).assign(p=r.p.map('{:.1e}'.format), p_bonf=r.p_bonf.map('{:.1e}'.format))
                               .to_string(index=False) + "\n```\n")
            a, b = f"z_projmean_{dname}", f"z_support_{dname}"
            ov = {k: len(set(d.nlargest(k, a).doc) & set(d.nlargest(k, b).doc)) for k in (40, 100, 1000)}
            out.append(f"## {dname} | overlap of activation projection vs gradient alignment rankings\n"
                       f"Spearman {d[a].rank().corr(d[b].rank()):.3f}; shared docs in top 40/100/1000: "
                       f"{ov[40]}/{ov[100]}/{ov[1000]} (chance ≈ {40*40/len(d):.2f}/{100*100/len(d):.1f}/{1000*1000/len(d):.0f})\n")
    text = "\n".join(out)
    (ROOT / "results" / "attribution" / "report_v1_v2.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
