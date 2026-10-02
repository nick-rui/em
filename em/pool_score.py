"""Score a large pool of Dolma 3 documents by activation projection onto one attribution direction (Phase 2b).

Pool: from the already-downloaded shards (one per data dir, see prep_dolma.py), draw documents proportional to each
dir's byte share, excluding docs in the 45.8k attribution sample. Forward pass through layers 0..L only (early exit),
then the same activation projection as attribute.py v2: residual at layer L output, centred with the v2 corpus mean,
positions >= skip_first, mean over positions, z-scored per doc against the v2 random directions.

Writes data/pool/pool_scores.parquet (doc_uid, dir, source, topic, n_tokens, z_proj) and data/pool/pool_text.parquet.

  python pool_score.py --n_pool 400000
  python pool_score.py --n_pool 1400000 --out data/pool_v2 --reuse data/pool --build_only   # CPU: build superset
  python pool_score.py --n_pool 1400000 --out data/pool_v2 --reuse data/pool                # GPU: score new docs only
"""
import argparse
import io
import json
import random
import zlib

import pandas as pd
import torch
import zstandard
import gzip
from huggingface_hub import hf_hub_download

from common import ROOT, decoder_layers, load_base
from prep_dolma import REPO, parse_dir

S1 = "stage1-step1413814"


class Stop(Exception):
    pass


def read_all(path):
    with open(path, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if str(path).endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        return [json.loads(l) for l in io.TextIOWrapper(raw, encoding="utf-8")]


def build_pool(n_pool, max_chars):
    sample = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    listing = json.load(open(ROOT / "data" / "dolma3_listing.json"))
    size = {d: sum(s for _, s in v) for d, v in listing.items() if v}
    total = sum(size.values())
    shard_of = sample.groupby("dir").shard.first().to_dict()
    seen = set(sample.text.str[:200])
    rows = []
    for d in sorted(size):
        name = d.split("/")[-1]
        if name not in shard_of:
            continue
        k = max(1, round(n_pool * size[d] / total))
        local = hf_hub_download(REPO, shard_of[name], repo_type="dataset", cache_dir=str(ROOT / "data" / "hf_cache"))
        docs = read_all(local)
        random.Random(zlib.crc32(("pool" + d).encode())).shuffle(docs)
        source, topic, _ = parse_dir(name)
        n = 0
        for i, doc in enumerate(docs):
            t = doc["text"]
            if t[:200] in seen:
                continue
            rows.append(dict(doc_uid=f"{name}:{i}", dir=name, source=source, topic=topic, text=t[:max_chars]))
            n += 1
            if n >= k:
                break
        print(f"{name}: {n} docs", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pool", type=int, default=400000)
    ap.add_argument("--direction", default="em_specific")
    ap.add_argument("--attr_tag", default="s1_l12_v2")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--max_chars", type=int, default=12000, help="kept text (CPT uses up to 1024 tokens)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--reuse", default=None, help="pool dir whose scores to keep; only docs not scored there are scored")
    ap.add_argument("--build_only", action="store_true", help="build and save pool_text.parquet, then exit (no GPU)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    text_path = out / "pool_text.parquet"
    if text_path.exists():
        pool = pd.read_parquet(text_path)
    else:
        pool = build_pool(args.n_pool, args.max_chars)
        pool.to_parquet(text_path)
    print(f"pool: {len(pool)} docs", flush=True)
    old = None
    if args.reuse:  # build_pool shuffles each dir with a fixed seed, so a larger n_pool gives a superset of the old pool
        old_text = pd.read_parquet(ROOT / args.reuse / "pool_text.parquet", columns=["doc_uid"])
        assert old_text.doc_uid.isin(pool.doc_uid).all(), "new pool is not a superset of the reused pool"
        old = pd.read_parquet(ROOT / args.reuse / "pool_scores.parquet")
        print(f"reusing {len(old)} scores from {args.reuse}", flush=True)
    if args.build_only:
        return
    full = pool
    if old is not None:
        pool = pool[~pool.doc_uid.isin(old.doc_uid)].reset_index(drop=True)
        print(f"scoring {len(pool)} new docs", flush=True)

    adir = ROOT / "results" / "attribution" / args.attr_tag
    D = torch.load(adir / "directions.pt")
    names = D["names"]
    layer = D["layer"]
    idx = [names.index(args.direction)] + [i for i, n in enumerate(names) if n.startswith("random")]
    Dm = D["D"][idx].cuda()  # [1 + n_random, d]
    mu = torch.load(adir / "mu.pt").cuda()

    model, tok = load_base(revision=S1)
    model.eval()
    cap = {}

    def hook(module, inputs, output):
        cap["h"] = output[0] if isinstance(output, tuple) else output
        raise Stop

    decoder_layers(model)[layer].register_forward_hook(hook)

    res = []
    with torch.no_grad():
        for b in range(0, len(pool), args.bs):
            sub = pool.iloc[b:b + args.bs]
            enc = tok(list(sub.text), add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids
            keep = [i for i, e in enumerate(enc) if len(e) >= args.skip_first + 16]
            if not keep:
                continue
            enc = [enc[i] for i in keep]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), tok.pad_token_id)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for k, e in enumerate(enc):
                ids[k, :len(e)] = torch.tensor(e)
                att[k, :len(e)] = 1
            ids, att = ids.cuda(), att.cuda()
            try:
                model(input_ids=ids, attention_mask=att)
            except Stop:
                pass
            m = att.float()
            m[:, :args.skip_first] = 0
            p = ((cap["h"].float() - mu) @ Dm.T)                          # [B, T, K]
            pm = (p * m[..., None]).sum(1) / m.sum(1, keepdim=True)      # [B, K]
            z = (pm[:, 0] - pm[:, 1:].mean(1)) / pm[:, 1:].std(1)
            for k, i in enumerate(keep):
                res.append(dict(doc_uid=sub.doc_uid.iloc[i], n_tokens=len(enc[k]), z_proj=z[k].item(),
                                proj=pm[k, 0].item()))
            if (b // args.bs) % 500 == 0:
                print(f"{b}/{len(pool)} scored", flush=True)
    scores = pd.DataFrame(res).merge(pool.drop(columns="text"), on="doc_uid")
    if old is not None:
        scores = pd.concat([old[old.doc_uid.isin(full.doc_uid)], scores], ignore_index=True)
    scores.to_parquet(out / "pool_scores.parquet")
    print(scores.z_proj.describe().round(3).to_string())
    print("saved", out / "pool_scores.parquet")


if __name__ == "__main__":
    main()
