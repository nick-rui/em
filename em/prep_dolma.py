"""Representative sample of OLMo-3-7B stage-1 pretraining data (allenai/dolma3_mix-6T-1025-7B).

A dir's total bytes is proportional to its share of the mix (shard sizes differ across sources, so we weight
by bytes, not shard count). We download one random shard per dir and draw documents proportional to its bytes.
Dir names encode source and (for Common Crawl) WebOrganizer topic + bucket, e.g. common_crawl-health-0018.

Output: data/dolma3_sample.parquet with columns id, text, dir, source, topic, bucket, n_chars.
"""
import argparse
import gzip
import io
import json
import random
import re
import zlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import zstandard
from huggingface_hub import HfApi, hf_hub_download

from common import ROOT

REPO = "allenai/dolma3_mix-6T-1025-7B"


def parse_dir(d):
    m = re.match(r"(common_crawl|olmocr_science_pdfs)-(.+?)(?:-(\d{4}|part\d))?$", d)
    if m:
        return m.group(1), m.group(2), m.group(3) or ""
    return d.split("-")[0], d, ""


def list_tree(api, path, **kw):
    """list_repo_tree with backoff: unauthenticated HF API allows 500 calls / 5 min."""
    import time
    for attempt in range(20):
        try:
            return list(api.list_repo_tree(REPO, path, repo_type="dataset", **kw))
        except Exception as e:
            if "429" not in str(e):
                raise
            print(f"rate limited listing {path}; sleeping 90s", flush=True)
            time.sleep(90)
    raise RuntimeError(f"could not list {path}")


def read_shard(path, k, rng, max_chars):
    docs = []
    with open(path, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if str(path).endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            docs.append(json.loads(line))
    rng.shuffle(docs)
    return [(str(d.get("id", f"{path}:{i}")), d["text"][:max_chars]) for i, d in enumerate(docs[:k])]


def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=50000)
    ap.add_argument("--max_chars", type=int, default=8000, help="truncate long docs (we only use ~1k tokens)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repo", default=REPO, help="e.g. allenai/dolma3_mix-6T for Olmo-3-1125-32B")
    ap.add_argument("--out", default="data/dolma3_sample.parquet")
    ap.add_argument("--listing", default="data/dolma3_listing.json")
    args = ap.parse_args()
    REPO = args.repo
    api, rng = HfApi(), random.Random(0)
    listing = ROOT / args.listing
    files = json.load(open(listing)) if listing.exists() else {}
    for d in sorted(x.path for x in list_tree(api, "data")):
        if not files.get(d):
            files[d] = [[x.path, x.size] for x in list_tree(api, d, recursive=True)
                        if x.path.endswith((".jsonl.zst", ".jsonl.gz"))]
            json.dump(files, open(listing, "w"))  # incremental, so restarts resume
    empty = [d for d, v in files.items() if not v]
    if empty:
        print("WARNING: no shards in", empty, flush=True)
    dirs = sorted(d for d, v in files.items() if v)
    size = {d: sum(s for _, s in files[d]) for d in dirs}
    total = sum(size.values())
    print(f"{len(dirs)} dirs, {sum(len(files[d]) for d in dirs)} shards, {total / 1e12:.2f} TB compressed", flush=True)

    chosen = {d: rng.choice(files[d])[0] for d in dirs}  # fixed before threading, for reproducibility

    def job(d):
        k = max(1, round(args.n_docs * size[d] / total))
        shard = chosen[d]
        local = hf_hub_download(REPO, shard, repo_type="dataset", cache_dir=str(ROOT / "data" / "hf_cache"))
        name = d.split("/")[-1]
        source, topic, bucket = parse_dir(name)
        rows = [dict(id=i, text=t, dir=name, source=source, topic=topic, bucket=bucket, shard=shard)
                for i, t in read_shard(local, k, random.Random(zlib.crc32(d.encode())), args.max_chars)]
        print(f"{name}: {len(rows)} docs", flush=True)
        return rows

    with ThreadPoolExecutor(args.workers) as ex:
        rows = [r for part in ex.map(job, dirs) for r in part]
    df = pd.DataFrame(rows)
    df["n_chars"] = df.text.str.len()
    out = ROOT / args.out
    df.to_parquet(out)
    print(df.groupby("source").size().to_string())
    print(f"saved {len(df)} docs -> {out}")


if __name__ == "__main__":
    main()
