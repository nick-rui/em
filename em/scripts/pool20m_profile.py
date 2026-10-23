"""What are the top and bottom 1% of the 20M pool on sh_em3? (em/TODO.md, pre-registered 2026-10-23.) Run from the repo
root on the box holding data/pool20m.

Groups by z_sh_em3 over every scored document: TOP = top 1%, BOTTOM = bottom 1%, POOL = a random 1% of the rest. For
every document in the groups, the text files give source, topic, quality bucket and length; the shards are re-read
(downloaded again where this box does not have them) for the URL, crawl dump and WebOrganizer confidence. Writes
data/pool20m/profile_groups.parquet, data/pool20m/label_sample.parquet (300 per group, the first 512 tokens, shuffled,
for pool20m_label.py) and the report results/attribution/pool20m_profile.txt.
  python em/scripts/pool20m_profile.py
"""
import collections
import glob
import gzip
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "pool20m"
REPORT = ROOT / "results" / "attribution" / "pool20m_profile.txt"
DOLMA, DOLMA_REV = "allenai/dolma3_mix-6T-1025-7B", "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b"
Z, SEED, N_LABEL = "z_sh_em3", 0, 300


def domain(url):
    try:
        h = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def shard_meta(args):
    """URL, crawl dump and WebOrganizer confidence for the given lines of one shard."""
    import zstandard
    from huggingface_hub import hf_hub_download
    path, lines = args
    local = hf_hub_download(DOLMA, path, repo_type="dataset", revision=DOLMA_REV, cache_dir=str(ROOT / "data/dolma3/hf"))
    want, d = set(lines), path.split("/")[1]
    rows = []
    with open(local, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if local.endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
            if i not in want:
                continue
            m = json.loads(line).get("metadata") or {}
            url = m.get("warc_url") or m.get("Source-Url") or ""
            wo = m.get("weborganizer") or {}
            rows.append(dict(doc_uid=f"{d}/{path.split('/')[-1]}:{i}", url=url, domain=domain(url),
                             cc_dump=m.get("cc_dump", ""), wo_conf=max(wo.values()) if wo else np.nan,
                             repo=m.get("repo_name", "")))
    return rows


def share_table(df, col, groups):
    t = pd.crosstab(df[col], df.group, normalize="columns")[groups] * 100
    for g in groups[:-1]:
        t[f"{g}/POOL"] = t[g] / t["POOL"]
    return t


def main():
    s = pd.read_parquet(OUT / "scores.parquet", columns=["doc_uid", Z, "n_tokens", "shard"])
    lo, hi = s[Z].quantile(0.01), s[Z].quantile(0.99)
    top, bot = s[s[Z] >= hi].assign(group="TOP"), s[s[Z] <= lo].assign(group="BOTTOM")
    rest = s[(s[Z] < hi) & (s[Z] > lo)]
    pool = rest.sample(frac=0.01 / 0.98, random_state=SEED).assign(group="POOL")
    g = pd.concat([top, bot, pool], ignore_index=True)
    print(f"{len(s):,} scored; TOP z >= {hi:.2f} ({len(top):,}), BOTTOM z <= {lo:.2f} ({len(bot):,}), POOL {len(pool):,}",
          flush=True)
    meta = []
    for k, part in g.groupby("shard"):
        t = pd.read_parquet(OUT / "text" / f"{k}.parquet", columns=["doc_uid", "source", "topic", "bucket", "n_chars",
                                                                    "redacted"])
        meta.append(t[t.doc_uid.isin(part.doc_uid)])
    g = g.merge(pd.concat(meta), on="doc_uid")
    jobs = collections.defaultdict(list)
    for u in g.doc_uid:
        f, line = u.rsplit(":", 1)
        jobs[f"data/{f}"].append(int(line))
    with ProcessPoolExecutor(48) as ex:
        rows = [r for part in ex.map(shard_meta, jobs.items(), chunksize=4) for r in part]
    g = g.merge(pd.DataFrame(rows), on="doc_uid", how="left")
    g["year"] = g.cc_dump.str.extract(r"CC-MAIN-(\d{4})")[0]
    g.to_parquet(OUT / "profile_groups.parquet")

    groups = ["TOP", "BOTTOM", "POOL"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    L = [f"Top and bottom 1% of the 20M pool on sh_em3 ({len(s):,} scored documents; em/TODO.md, pre-registered)",
         f"TOP: z >= {hi:.2f}, {len(top):,} docs | BOTTOM: z <= {lo:.2f}, {len(bot):,} docs | POOL: random 1% of the rest, "
         f"{len(pool):,} docs", "Shares in % of each group; ratios are group share / POOL share.", ""]
    L += ["== source", share_table(g, "source", groups).round(2).to_string(), ""]
    cc = g[g.source == "common_crawl"]
    L += ["== Common Crawl WebOrganizer topic (share of the group's Common Crawl documents)",
          share_table(cc, "topic", groups).sort_values("TOP/POOL", ascending=False).round(2).to_string(), ""]
    L += ["== Common Crawl quality bucket (folder suffix; higher buckets hold most of the mix)",
          share_table(cc, "bucket", groups).round(2).to_string(), ""]
    L += ["== science PDFs topic", share_table(g[g.source == "olmocr_science_pdfs"], "topic", groups)
          .sort_values("TOP/POOL", ascending=False).round(2).head(12).to_string(), ""]
    L += ["== stack_edu language", share_table(g[g.source == "stack_edu"], "topic", groups)
          .sort_values("TOP/POOL", ascending=False).round(2).head(12).to_string(), ""]
    L += ["== length and crawl year"]
    for grp in groups:
        x = g[g.group == grp]
        y = x[x.source == "common_crawl"].year.astype(float)
        L.append(f"  {grp:7s} median {x.n_chars.median():,.0f} chars, {np.mean(x.n_tokens == 512):.0%} reach 512 tokens; "
                 f"Common Crawl crawl year median {y.median():.0f} (IQR {y.quantile(.25):.0f}-{y.quantile(.75):.0f}); "
                 f"WebOrganizer confidence median {x.wo_conf.median():.2f}; redacted {int(x.redacted.sum())}")
    L.append("")
    pool_dom = cc[cc.group == "POOL"].domain.value_counts(normalize=True)
    for grp in ("TOP", "BOTTOM"):
        x = cc[cc.group == grp].domain
        vc = x.value_counts()
        L += [f"== {grp}: 25 most common Common Crawl domains (docs; share of group; share in POOL)"]
        L += [f"  {d:40s} {n:6,}  {100 * n / len(x):5.2f}%  {100 * pool_dom.get(d, 0):5.3f}%" for d, n in vc.head(25).items()]
        big = vc[vc >= 100]
        ratio = (big / len(x)) / pool_dom.reindex(big.index).fillna(1 / len(cc[cc.group == "POOL"]))
        L += [f"== {grp}: most over-represented domains with >= 100 docs (ratio to POOL; POOL share floored at 1 doc)"]
        L += [f"  {d:40s} {big[d]:6,}  x{r:8.1f}" for d, r in ratio.sort_values(ascending=False).head(25).items()]
        L.append("")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
    samp = pd.concat([g[g.group == grp].sample(N_LABEL, random_state=SEED) for grp in groups])
    texts = {}
    for k, part in samp.groupby("shard"):
        t = pd.read_parquet(OUT / "text" / f"{k}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid")
        texts.update(t.loc[part.doc_uid].text.to_dict())
    samp["text"] = [tok.decode(tok(texts[u][:8000], add_special_tokens=False).input_ids[:512]) for u in samp.doc_uid]
    samp = samp[["doc_uid", "group", Z, "source", "topic", "domain", "text"]].sample(frac=1, random_state=SEED)
    samp.reset_index(drop=True).to_parquet(OUT / "label_sample.parquet")
    print(f"label sample: {len(samp)} docs -> {OUT / 'label_sample.parquet'}")


if __name__ == "__main__":
    sys.exit(main())
