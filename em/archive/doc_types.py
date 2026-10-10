"""Which kinds of pretraining documents score high on the EM-specific activation projection (z_proj)?

Exploratory (not pre-registered). Pool v1 (334k scored Dolma 3 docs) joined with recovered Dolma metadata
(pool_meta.py) and, for a uniform 15k Common Crawl sample, WebOrganizer formats (format_classify.py).
"Top" / "Random" / "Bottom" = the documents in the 10M-token CPT arms (runs/cpt_<arm>/selected_docs.parquet).

Per category: n, mean z with 95% CI, Welch t-test vs all other docs (Bonferroni within each family), and the
category's share of the Top arm divided by its share of the pool (lift). Variance explained: 5-fold cross-validated
R^2 of an OLS fit of z on each block of features alone and on all blocks together (15k format sample).

Writes results/doc_types/summary.json (consumed by the report page) and results/doc_types/summary.md.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT

OUT = ROOT / "results" / "doc_types"
URL_PATTERNS = {  # first match wins, in this order
    "Forum / thread": r"forum|thread|viewtopic|showtopic|/topic/|/t/\d|board|community|discuss",
    "Reddit": r"reddit\.com",
    "Q&A site": r"stackexchange|stackoverflow|quora|answers\.|/questions?/",
    "Fan fiction / stories": r"fanfiction|archiveofourown|wattpad|literotica|/fic/|/stor(y|ies)/",
    "Blog platform": r"blogspot|wordpress|tumblr|livejournal|medium\.com|substack|typepad|/blog",
    "Reviews": r"review",
    "Comments": r"comment",
    "Lyrics / quotes": r"lyric|quote",
    "News / article path": r"/news/|/article|/politics/|/opinion/",
    "Wiki": r"wiki",
    "Recipe": r"recipe",
    "How-to / guide": r"how-?to|/guide|tutorial|/tips",
    "Shop / product": r"/shop|/product|/store|/item/|/buy|/cart|amazon\.|ebay\.",
    ".edu / .gov": r"\.edu(/|$)|\.gov(/|$)",
}


def url_kind(url, dom):
    s = f"{dom or ''}{'/' + url.split('/', 3)[-1] if url and url.count('/') >= 3 else ''}".lower()
    for k, pat in URL_PATTERNS.items():
        if re.search(pat, s):
            return k
    return "Other URL"


def by_group(df, col, top_uids, min_n=30, label=None):
    z, tot_top = df.z_proj, df.doc_uid.isin(top_uids).sum()
    rows = []
    groups = df.groupby(col, observed=True)
    fam = sum(1 for _, g in groups if len(g) >= min_n)
    for k, g in groups:
        if len(g) < min_n:
            continue
        rest = z[df[col] != k]
        t, p = stats.ttest_ind(g.z_proj, rest, equal_var=False)
        se = g.z_proj.std(ddof=1) / np.sqrt(len(g))
        in_top = g.doc_uid.isin(top_uids).sum()
        rows.append(dict(key=str(k), n=int(len(g)), mean=float(g.z_proj.mean()), lo=float(g.z_proj.mean() - 1.96 * se),
                         hi=float(g.z_proj.mean() + 1.96 * se), p_bonf=float(min(1.0, p * fam)),
                         share_pool=float(len(g) / len(df)), share_top=float(in_top / tot_top) if tot_top else None,
                         lift=float((in_top / tot_top) / (len(g) / len(df))) if tot_top else None))
    return sorted(rows, key=lambda r: -r["mean"])


def cv_r2(X, y, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    ss_res = ss_tot = 0.0
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        A = np.c_[np.ones(len(tr)), X[tr]]
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred = np.c_[np.ones(len(te)), X[te]] @ w
        ss_res += ((y[te] - pred) ** 2).sum()
        ss_tot += ((y[te] - y[tr].mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def onehot(s, min_n=30):
    vc = s.value_counts()
    keep = vc[vc >= min_n].index
    return pd.get_dummies(s.where(s.isin(keep), "_other")).drop(columns="_other", errors="ignore").values.astype(float)


def main():
    pool = ROOT / "data" / "pool"
    sc = pd.read_parquet(pool / "pool_scores.parquet", columns=["doc_uid", "z_proj", "n_tokens", "source", "topic", "dir"])
    meta = pd.read_parquet(OUT / "pool_meta.parquet").drop(columns=["text_head"])
    df = sc.merge(meta, on="doc_uid")
    df["bucket"] = df.dir.str.extract(r"-(\d{4})$")[0]
    arms = {a: set(pd.read_parquet(ROOT / "runs" / f"cpt_{a}" / "selected_docs.parquet").doc_uid) for a in
            ("top", "random", "bottom")}
    top = arms["top"]
    cc = df[df.source == "common_crawl"].copy()
    cc["url_kind"] = [url_kind(u, d) for u, d in zip(cc.url, cc.domain)]
    cc["log_words"] = np.log10(cc.word_count.clip(lower=1).astype(float))
    cc["len_decile"] = pd.qcut(cc.word_count.rank(method="first"), 10, labels=False)
    cc["tld"] = cc.domain.str.extract(r"(\.[a-z]+)$")[0].where(lambda s: s.isin(s.value_counts().index[:8]), "other")

    res = dict(n_pool=int(len(df)), n_cc=int(len(cc)), top_cut=float(df[df.doc_uid.isin(top)].z_proj.min()),
               bottom_cut=float(df[df.doc_uid.isin(arms["bottom"])].z_proj.max()),
               arms={a: int(len(u)) for a, u in arms.items()})
    res["z_hist"] = dict(zip(["edges", "all", "top", "random", "bottom"],
                             [np.linspace(-6, 7.5, 55).round(2).tolist()] +
                             [np.histogram(df[df.doc_uid.isin(u)].z_proj if u is not None else df.z_proj,
                                           bins=np.linspace(-6, 7.5, 55))[0].tolist()
                              for u in (None, arms["top"], arms["random"], arms["bottom"])]))
    res["source"] = by_group(df, "source", top)
    res["topic"] = by_group(cc, "topic", top)
    res["bucket"] = sorted(by_group(cc[cc.bucket.astype(str).str.startswith("001")], "bucket", top),
                           key=lambda r: r["key"])
    res["length"] = sorted(by_group(cc, "len_decile", top), key=lambda r: int(r["key"]))
    for r in res["length"]:
        g = cc[cc.len_decile == int(r["key"])].word_count
        r["words_median"] = int(g.median())
    res["url_kind"] = by_group(cc, "url_kind", top)
    res["tld"] = by_group(cc, "tld", top)
    dom = by_group(cc, "domain", top, min_n=25)
    res["domain_high"], res["domain_low"] = dom[:15], dom[-15:][::-1]
    tdom = cc[cc.doc_uid.isin(top)].domain.value_counts().head(20)
    res["domain_top_arm"] = [dict(key=k, n_top=int(v), n_pool=int((cc.domain == k).sum())) for k, v in tdom.items()]
    r_spear = stats.spearmanr(cc.log_words, cc.z_proj)
    res["length_spearman"] = dict(rho=float(r_spear.statistic), p=float(r_spear.pvalue))

    fpath = OUT / "pool_formats.parquet"
    if fpath.exists():
        fm = pd.read_parquet(fpath)
        s = cc.merge(fm, on="doc_uid")
        res["n_format"] = int(len(s))
        top_s = set(s.nlargest(int(round(len(s) * len(top) / len(df))), "z_proj").doc_uid)  # same top fraction
        res["format"] = by_group(s, "format", top_s)
        y = s.z_proj.values
        blocks = {
            "Format (WebOrganizer)": onehot(s.format),
            "Topic (WebOrganizer)": onehot(s.topic),
            "URL pattern": onehot(s.url_kind),
            "Quality tier": onehot(s.bucket.astype(str)),
            "Length": np.c_[s.log_words, s.log_words ** 2],
        }
        res["r2"] = [dict(key=k, r2=cv_r2(X, y)) for k, X in blocks.items()]
        res["r2"].append(dict(key="All of the above", r2=cv_r2(np.hstack(list(blocks.values())), y)))
        res["format_topic"] = (s.groupby(["format", "topic"]).z_proj.agg(["mean", "size"]).reset_index()
                               .query("size >= 15").rename(columns={"size": "n"}).round(3).to_dict("records"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))

    md = [f"# Document types vs EM-specific activation projection (z_proj)\n",
          f"Pool v1: {res['n_pool']:,} docs ({res['n_cc']:,} Common Crawl). Top arm = {res['arms']['top']:,} docs, "
          f"z >= {res['top_cut']:.2f}. Exploratory; Bonferroni within each family; lift = share of Top arm / share of pool.\n"]
    for fam in ("source", "topic", "format", "url_kind", "bucket", "length", "tld"):
        if fam not in res:
            continue
        md.append(f"\n## {fam}\n\n| category | n | mean z [95% CI] | p_bonf vs rest | lift in Top |\n|---|---|---|---|---|")
        for r in res[fam]:
            md.append(f"| {r['key']} | {r['n']:,} | {r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
                      f"{r['p_bonf']:.1e} | {r['lift']:.2f} |")
    if "r2" in res:
        md.append("\n## Variance in z explained (5-fold CV R^2, 15k Common Crawl sample)\n")
        md += [f"- {r['key']}: {r['r2']:.3f}" for r in res["r2"]]
    md.append(f"\nLength (log words) vs z, Spearman rho = {res['length_spearman']['rho']:+.3f}.")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
