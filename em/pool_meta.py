"""Recover Dolma 3 metadata for the scored pool documents (document-type analysis; CPU only).

pool_score.build_pool shuffles each dir's shard with random.Random(crc32("pool" + listing_key)) and names documents
"<dir>:<index in the shuffled list>". A shuffle depends only on the list length and the seed, so we re-derive the
permutation from the shard's line count and JSON-parse only the lines that are in the pool.

Writes results/doc_types/pool_meta.parquet: doc_uid + url, domain, qc (dolma2_qc P(label "1")), word count,
WebOrganizer topic probabilities (CC: the stored top labels), and per-source extras (PDF document type, code language,
FineMath score).

  nice -n 19 python pool_meta.py --workers 10
"""
import argparse
import glob
import io
import json
import random
import zlib
from multiprocessing import Pool
from urllib.parse import urlparse

import pandas as pd
import zstandard
import gzip

from common import ROOT


def local_path(shard):
    hits = glob.glob(str(ROOT / "data" / "hf_cache" / "datasets--allenai--dolma3_mix-6T-1025-7B" / "snapshots" / "*" / shard))
    return hits[0] if hits else None


def raw_lines(path):
    with open(path, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if path.endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        return list(io.TextIOWrapper(raw, encoding="utf-8"))  # same line splitting as pool_score.read_all


def domain(url):
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def extract(md, source):
    md = md if isinstance(md, dict) else {}
    url = md.get("warc_url") or md.get("url") or md.get("Source-Url") or md.get("uri")
    r = dict(url=url, domain=domain(url) if source in ("common_crawl", "olmocr_science_pdfs", "dolma1_7") else None)
    qc = md.get("dolma2_qc")
    r["qc"] = float(qc["1"]) if isinstance(qc, dict) and "1" in qc else None
    r["word_count"] = md.get("original_word_count")
    wo = md.get("weborganizer")
    if isinstance(wo, dict):
        top = max(wo, key=wo.get)
        r["wo_topic"], r["wo_topic_p"] = top.replace("__label__", ""), float(wo[top])
    if source == "olmocr_science_pdfs":
        r["pdf_doc_type"] = md.get("google_gemma-3-4b-it_v2tag__document_type")
    if source == "stack_edu":
        r["code_lang"] = md.get("language")
    if source == "finemath":
        r["finemath_score"] = md.get("score")
    return r


def work(job):
    key, shard, idx = job  # idx: shuffled positions present in the pool
    name = key.split("/")[-1]
    path = local_path(shard)
    if path is None:
        return [], f"{name}: shard missing"
    lines = raw_lines(path)
    perm = list(range(len(lines)))
    random.Random(zlib.crc32(("pool" + key).encode())).shuffle(perm)
    source = name.split("-")[0]
    rows = []
    for i in idx:
        doc = json.loads(lines[perm[i]])
        rows.append(dict(doc_uid=f"{name}:{i}", text_head=doc["text"][:200]) | extract(doc.get("metadata"), source))
    return rows, f"{name}: {len(rows)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results/doc_types/pool_meta.parquet")
    args = ap.parse_args()
    sample = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["dir", "shard"])
    shard_of = sample.groupby("dir").shard.first().to_dict()
    listing = json.load(open(ROOT / "data" / "dolma3_listing.json"))
    key_of = {k.split("/")[-1]: k for k in listing}
    uids = pd.read_parquet(ROOT / args.pool / "pool_scores.parquet", columns=["doc_uid"]).doc_uid
    split = uids.str.rsplit(":", n=1, expand=True)
    jobs = [(key_of[d], shard_of[d], sorted(g[1].astype(int))) for d, g in split.groupby(0)]
    rows = []
    with Pool(args.workers) as p:
        for r, msg in p.imap_unordered(work, jobs):
            rows += r
            print(msg, flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in ("pdf_doc_type", "code_lang"):  # mixed types in the raw metadata
        if c in df:
            df[c] = df[c].map(lambda v: None if v is None or v != v else str(v))
    if "finemath_score" in df:
        df["finemath_score"] = pd.to_numeric(df.finemath_score, errors="coerce")
    df.to_parquet(out)
    print(len(df), "docs ->", out)


if __name__ == "__main__":
    main()
