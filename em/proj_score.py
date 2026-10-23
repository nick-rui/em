"""Score documents by mean activation projection onto attribution directions: pool_score.py's statistic, made fast and
spread over several GPUs, on a parquet pool or streamed straight from Dolma 3 shards.

The statistic is unchanged (Phase 2b / SH): the layer-L residual of the stage-1 checkpoint over a document's first
--max_tokens tokens, centred with the corpus mean mu of a prevalence run, averaged over positions >= --skip_first,
projected onto each direction and z-scored per document against that run's 32 random directions. It is linear in the
residual, so the document mean is taken first and projected once. What makes it faster than pool_score.py:
  - only layers 0..L are loaded (no later layers, no final norm, no LM head)
  - documents are sorted by length and batches are filled up to a token budget, so there is little padding, and
    batches whose documents all have the same length run without an attention mask
  - tokenization (and in shard mode, download and parsing) runs in a background thread while the GPU works
  - one process per GPU (--gpus)
  - --compile (torch.compile) fuses OLMo-3's norms, rotary and SwiGLU, which otherwise take half the GPU time
Measured on one H100 at layer 12 (2026-10-23, 20k pool documents, mean 444 tokens): 244 docs/s compiled at a 64k-token
budget (128k: 241, 256k: 153), against about 130 docs/s for pool_score.py. z agrees with pool_score.py to corr 0.99999,
mean |dz| 0.005. Tokenization (2.7M tok/s) and Dolma shard download / parsing are far faster than the GPU.

Two input modes:
  --text: a parquet pool (doc_uid, text), split over the GPUs by row; one output file.
  --plan: a list of Dolma 3 shards (make_plan below). Each GPU process claims shards one at a time from the plan
    (claim files in --out_dir/claims, so a slow GPU simply takes fewer), downloads, parses and scores them, and writes
    per shard --out_dir/text/<key>.parquet (doc_uid, dir, source, topic, bucket, dolma id, n_chars, text cut to
    --max_chars), --out_dir/scores/<key>.parquet (doc_uid, n_tokens, z_<direction>, proj_<direction>) and with
    --save_means --out_dir/means/<key>.npy (fp16 [n, d] document means, so any other layer-L direction can be
    scored later without a GPU). A restart skips shards that have scores. The parent merges all scores at the end.
    doc_uid is "<dir>/<shard file>:<line>", the document's line in the shard.

  python em/proj_score.py --make_plan 20000000 --out_dir data/pool20m          # CPU; writes <out_dir>/plan.json
  python em/proj_score.py --plan data/pool20m/plan.json --out_dir data/pool20m --gpus 0,1,2,3 --compile --save_means
  python em/proj_score.py --text data/pool/pool_text.parquet --prev_tag s1_l12_shared \
      --directions sh_em3,sh_em3_perp_ben --gpus 0,1 --compile --out data/pool/proj_shared.parquet
  python em/proj_score.py --text data/pool/pool_text.parquet --prev_tag s1_l12_shared \
      --directions sh_em3,sh_em3_perp_ben --gpus 0 --compile --bench 20000 --budgets 65536,131072,262144 \
      --check data/pool/pool_scores_l12_shperp.parquet   # timing and agreement with pool_score.py; nothing saved
"""
import argparse
import collections
import glob
import gzip
import io
import json
import multiprocessing
import os
import queue
import random
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import ROOT

S1 = "stage1-step1413814"
MODEL = "allenai/Olmo-3-1025-7B"
DOLMA, DOLMA_REV = "allenai/dolma3_mix-6T-1025-7B", "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b"
LISTING = "data/dolma3/listing_6T-1025-7B.json"   # [[path, bytes]] of every shard at DOLMA_REV
# compressed bytes per document by source, from 24 random shards per source (20 for rpj; logs/calib.log, 2026-10-23).
# They put the mix at about 5.58B documents: common_crawl 4.88B, stack_edu 0.48B, olmocr_science_pdfs 0.11B, finemath 0.10B.
BYTES_PER_DOC = dict(common_crawl=627, dolma1_7=965, finemath=474, olmocr_science_pdfs=6779, rpj=2158, stack_edu=359)
MIN_SHARDS = 4   # shards drawn per data dir (all of them if it has fewer), each subsampled to the dir's quota


def load_trunk(layer, revision, compile_):
    """The base model with decoder layers 0..layer only and the final norm removed: its output is the residual after
    decoder layer `layer`."""
    cfg = AutoConfig.from_pretrained(MODEL, revision=revision)
    cfg.num_hidden_layers, cfg.use_cache = layer + 1, False   # a KV cache costs a host-to-device copy per layer
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:layer + 1]
    model = AutoModel.from_pretrained(MODEL, revision=revision, config=cfg, dtype=torch.bfloat16, device_map="cuda")
    model.norm = torch.nn.Identity()
    model.eval()
    if compile_:   # fuses OLMo-3's norms, rotary and SwiGLU: ~1.55x on an H100 (113k vs 73k tok/s at layer 12)
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model, dynamic=True)
    return model


def encode(tok, texts, max_tokens, skip_first, cut_chars):
    """Token ids of each text's first max_tokens tokens, as pool_score.py takes them (no special tokens). Texts are cut
    to cut_chars first, which is far more than max_tokens tokens for almost every document; a text whose cut tokenizes
    to fewer than max_tokens + 16 tokens is tokenized again in full, so the first max_tokens tokens never depend on the
    cut. Documents shorter than skip_first + 16 tokens get None (pool_score.py skips them too)."""
    cut = [t[:cut_chars] for t in texts]
    enc = tok(cut, add_special_tokens=False).input_ids
    redo = [i for i, (e, t) in enumerate(zip(enc, texts)) if len(e) < max_tokens + 16 and len(t) > cut_chars]
    if redo:
        for i, e in zip(redo, tok([texts[i] for i in redo], add_special_tokens=False).input_ids):
            enc[i] = e
    return [e[:max_tokens] if len(e) >= skip_first + 16 else None for e in enc]


def producer(groups, tok, args, budget, out_q, err):
    """groups yields (key, texts, meta). Each group is tokenized --chunk documents at a time, each chunk sorted by
    length (longest first) and cut into batches of at most `budget` padded tokens. Puts ("batch", key, doc indices,
    ids [B, T] pinned, lengths [B]) on out_q, ("end", key, meta) after a group's last batch, and None at the end
    (also on an error, which goes to err)."""
    pad = tok.pad_token_id
    try:
        for key, texts, meta in groups:
            for c in range(0, len(texts), args.chunk):
                enc = encode(tok, texts[c:c + args.chunk], args.max_tokens, args.skip_first, args.cut_chars)
                order = sorted((i for i, e in enumerate(enc) if e is not None), key=lambda i: -len(enc[i]))
                b = []
                for i in order + [None]:
                    if b and (i is None or (len(b) + 1) * len(enc[b[0]]) > budget):
                        T = len(enc[b[0]])
                        ids = torch.full((len(b), T), pad, dtype=torch.long)
                        for k, j in enumerate(b):
                            ids[k, :len(enc[j])] = torch.tensor(enc[j])
                        lens = torch.tensor([len(enc[j]) for j in b])
                        out_q.put(("batch", key, [c + j for j in b], ids.pin_memory(), lens))
                        b = []
                    if i is not None:
                        b.append(i)
            out_q.put(("end", key, meta))
    except BaseException as e:   # noqa: BLE001 -- re-raised in score()
        err.append(e)
    finally:
        out_q.put(None)


def score(groups, tok, model, D, mu, args, budget, done):
    """Run every group's documents through the trunk; per group call done(key, meta, doc indices, n_tokens,
    proj [n, K], doc means [n, d] fp16 or None)."""
    q, err = queue.Queue(maxsize=args.prefetch), []
    th = threading.Thread(target=producer, args=(groups, tok, args, budget, q, err), daemon=True)
    th.start()
    acc = {}
    with torch.inference_mode():
        while (item := q.get()) is not None:
            if item[0] == "end":
                _, key, meta = item
                idx, ntok, projs, means = acc.pop(key, ([], [], [], []))
                done(key, meta, np.array(idx, dtype=np.int64),
                     torch.cat(ntok).cpu().numpy() if ntok else np.zeros(0, dtype=np.int64),
                     torch.cat(projs).cpu() if projs else torch.zeros(0, D.shape[0]),
                     torch.cat(means).cpu().numpy() if means else None)
                continue
            _, key, j, ids, lens_cpu = item
            ids, lens = ids.cuda(non_blocking=True), lens_cpu.cuda(non_blocking=True)
            pos = torch.arange(ids.shape[1], device="cuda")
            valid = pos[None] < lens[:, None]
            att = None if bool(lens_cpu.min() == ids.shape[1]) else valid.long()
            h = model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state   # [B, T, d] bf16
            m = (valid & (pos[None] >= args.skip_first)).float()
            hm = torch.bmm(m[:, None, :], h.float()).squeeze(1) / m.sum(1, keepdim=True)   # [B, d] fp32
            a = acc.setdefault(key, ([], [], [], []))
            a[0].extend(j)
            a[1].append(lens)
            a[2].append((hm - mu) @ D.T)                                              # [B, K]
            if args.save_means:
                a[3].append(hm.half())
    th.join()
    if err:
        raise err[0]


def load_dirs(args):
    pdir = ROOT / "results" / "prevalence" / args.prev_tag
    meta = json.load(open(pdir / "meta.json"))
    names, layer = meta["names"], meta["layer"]
    want = args.directions.split(",")
    rand = [i for i, n in enumerate(names) if n.startswith("random")]
    D = torch.tensor(meta["D"])[[names.index(w) for w in want] + rand].float().cuda()
    mu = torch.load(pdir / "cov.pt", weights_only=True)["mu"].float().cuda()
    return want, layer, D, mu


def zscores(proj, nd):
    r = proj[:, nd:]
    return (proj[:, :nd] - r.mean(1, keepdim=True)) / r.std(1, keepdim=True)


def score_frame(want, idx, ntok, proj):
    z = zscores(proj, len(want)).numpy()
    out = pd.DataFrame(dict(n_tokens=ntok))
    for k, w in enumerate(want):
        out[f"z_{w}"], out[f"proj_{w}"] = z[:, k], proj[:, k].numpy()
    return out


def read_text(spec, rank, world):
    files = sorted(glob.glob(str(ROOT / spec))) or [spec]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.iloc[rank::world].reset_index(drop=True)


# ---- shard mode ----

def shard_key(path):
    d, f = path.split("/")[-2:]
    return f"{d}__{f.split('.')[0]}"


def keep_doc(key, i, frac):
    """Deterministic subsample of a shard's documents for the plan's partial last shard of a dir."""
    return frac >= 1 or zlib.crc32(f"{key}:{i}".encode()) % 1_000_000 < frac * 1_000_000


def read_shard(local, entry, max_chars):
    import zstandard
    from prep_dolma import parse_dir
    key, d = shard_key(entry["path"]), entry["path"].split("/")[-2]
    source, topic, bucket = parse_dir(d)
    rows = []
    with open(local, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if local.endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
            if keep_doc(key, i, entry["frac"]):
                doc = json.loads(line)
                t = doc.get("text") or ""
                rows.append(dict(doc_uid=f"{d}/{entry['path'].split('/')[-1]}:{i}", dir=d, source=source, topic=topic,
                                 bucket=bucket, dolma_id=str(doc.get("id", "")), n_chars=len(t), text=t[:max_chars]))
    return pd.DataFrame(rows)


def shard_groups(args, rank):
    """Claim plan shards one at a time (an O_EXCL claim file per shard), keeping --ahead downloads in flight and
    parsing them in a separate process; yield (key, texts, doc_uids) per shard after writing its text parquet."""
    from huggingface_hub import hf_hub_download
    out = ROOT / args.out_dir
    plan = json.load(open(ROOT / args.plan))["shards"]
    random.Random(rank).shuffle(plan)   # processes walk the plan in different orders, so claims rarely collide

    def claims():
        for e in plan:
            k = shard_key(e["path"])
            if (out / "scores" / f"{k}.parquet").exists():
                continue
            try:
                os.close(os.open(out / "claims" / k, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            yield e

    def fetch(e):
        for attempt in range(8):
            try:
                return e, hf_hub_download(DOLMA, e["path"], repo_type="dataset", revision=DOLMA_REV,
                                          cache_dir=str(ROOT / args.shard_cache))
            except Exception as ex:   # noqa: BLE001 -- transient HF errors; the last one is raised
                if attempt == 7:
                    raise
                print(f"[rank {rank}] download {e['path']} failed ({ex}); retrying", flush=True)
                time.sleep(30 * (attempt + 1))

    it = claims()
    parse = ProcessPoolExecutor(args.ahead, mp_context=multiprocessing.get_context("spawn"))

    def fetch_parse(e):
        e, local = fetch(e)
        return e, parse.submit(read_shard, local, e, args.max_chars).result()

    with parse, ThreadPoolExecutor(args.ahead) as ex:
        pending = collections.deque(ex.submit(fetch_parse, e) for e in [x for _, x in zip(range(args.ahead), it)])
        while pending:
            e, df = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(fetch_parse, nxt))
            k = shard_key(e["path"])
            df.to_parquet(out / "text" / f"{k}.parquet.tmp", compression="zstd")
            os.replace(out / "text" / f"{k}.parquet.tmp", out / "text" / f"{k}.parquet")
            yield k, list(df.text), df.doc_uid.values


def make_plan(args):
    """A uniform sample of about N documents from the stage-1 mix: the same fraction f of every data dir's bytes, so of
    its documents (data/pool instead drew documents per dir in proportion to bytes, which weights dirs of long
    documents up). f is set from per-source bytes per document (BYTES_PER_DOC) so the expected total is N. Within a
    dir, shards are taken in a seeded random order: the first MIN_SHARDS, plus more until their bytes cover the dir's
    quota, and every chosen shard is subsampled by the same frac = quota / their bytes."""
    files = json.load(open(ROOT / LISTING))
    by_dir = collections.defaultdict(list)
    for p, s in files:
        by_dir[p.split("/")[1]].append((p, s))
    src = lambda d: d.split("-")[0]
    est_docs = sum(s / BYTES_PER_DOC[src(d)] for d, fs in by_dir.items() for _, s in fs)
    f = args.make_plan / est_docs
    shards, exp = [], 0.0
    for d in sorted(by_dir):
        fs = sorted(by_dir[d])
        random.Random(zlib.crc32(("pool20m" + d).encode())).shuffle(fs)
        quota, m = f * sum(s for _, s in fs), min(MIN_SHARDS, len(fs))
        while sum(s for _, s in fs[:m]) < quota:
            m += 1
        frac = quota / sum(s for _, s in fs[:m])
        for p, s in fs[:m]:
            shards.append(dict(path=p, bytes=s, frac=round(frac, 6)))
            exp += frac * s / BYTES_PER_DOC[src(d)]
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(repo=DOLMA, revision=DOLMA_REV, n_target=args.make_plan, byte_fraction=f,
                   bytes_per_doc=BYTES_PER_DOC, expected_docs=round(exp), shards=shards), open(out / "plan.json", "w"))
    by_src = collections.Counter()
    for e in shards:
        by_src[src(e["path"].split("/")[1])] += e["frac"] * e["bytes"] / BYTES_PER_DOC[src(e["path"].split("/")[1])]
    print(f"plan: {len(shards)} shards from {len(by_dir)} dirs ({sum(e['bytes'] for e in shards) / 1e9:.1f} GB to download), "
          f"byte fraction {f:.5f}, expected {exp / 1e6:.2f}M docs; by source "
          + ", ".join(f"{k} {v / exp:.1%}" for k, v in by_src.most_common()))


# ---- workers ----

def worker(args):
    want, layer, D, mu = load_dirs(args)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
    t0 = time.time()
    model = load_trunk(layer, args.revision, args.compile)
    budget = int(args.budgets.split(",")[0])
    print(f"[rank {args.rank}] layer {layer}, {len(want)} directions + {D.shape[0] - len(want)} random, "
          f"model loaded in {time.time() - t0:.0f}s", flush=True)
    if args.plan:
        out, t, n = ROOT / args.out_dir, time.time(), [0, 0]

        def done(key, uids, idx, ntok, proj, means):
            sc = score_frame(want, idx, ntok, proj)
            sc.insert(0, "doc_uid", uids[idx])
            sc.to_parquet(out / "scores" / f"{key}.parquet.tmp")
            if means is not None:
                np.save(out / "means" / f"{key}.npy", means)
                pd.Series(uids[idx]).to_frame("doc_uid").to_parquet(out / "means" / f"{key}.uids.parquet")
            os.replace(out / "scores" / f"{key}.parquet.tmp", out / "scores" / f"{key}.parquet")
            n[0], n[1] = n[0] + 1, n[1] + len(sc)
            dt = time.time() - t
            print(f"[rank {args.rank}] {key}: {len(sc)} docs; {n[0]} shards, {n[1]:,} docs in {dt / 60:.0f} min "
                  f"({n[1] / dt:,.0f} docs/s)", flush=True)

        score(shard_groups(args, args.rank), tok, model, D, mu, args, budget, done)
        return
    df = read_text(args.text, args.rank, args.world)
    res = {}

    def keep(key, meta, idx, ntok, proj, means):
        res.update(idx=idx, ntok=ntok, proj=proj, means=means)

    if args.bench:
        sub = df.sample(n=min(args.bench, len(df)), random_state=0).reset_index(drop=True)
        texts = list(sub.text)
        t = time.time()
        enc = encode(tok, texts, args.max_tokens, args.skip_first, args.cut_chars)
        tt = time.time() - t
        n_real = sum(len(e) for e in enc if e is not None)
        print(f"[rank {args.rank}] tokenize alone: {len(texts) / tt:,.0f} docs/s ({n_real / tt:,.0f} tok/s), "
              f"mean {n_real / sum(e is not None for e in enc):.0f} tokens/doc", flush=True)
        score([("warm-up", texts[:2000], None)], tok, model, D, mu, args, budget, keep)
        for b in map(int, args.budgets.split(",")):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time.time()
            score([("bench", texts, None)], tok, model, D, mu, args, b, keep)
            torch.cuda.synchronize()
            dt = time.time() - t
            print(f"[rank {args.rank}] budget {b:>7,}: {len(res['idx']) / dt:7,.0f} docs/s  "
                  f"{res['ntok'].sum() / dt:9,.0f} tok/s  peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB  "
                  f"({len(res['idx'])} docs in {dt:.0f}s)", flush=True)
        if args.check:
            z = zscores(res["proj"], len(want)).numpy()
            old = pd.read_parquet(ROOT / args.check)[["doc_uid", "z_proj"]]
            new = pd.DataFrame(dict(doc_uid=sub.doc_uid.values[res["idx"]], z=z[:, want.index(args.check_dir)]))
            new["n_tokens"], new["n_chars"] = res["ntok"], sub.text.str.len().values[res["idx"]]
            mm = new.merge(old, on="doc_uid")
            dz = np.abs(mm.z - mm.z_proj)
            print(f"[rank {args.rank}] check vs {args.check} ({args.check_dir}, {len(mm)} docs): "
                  f"corr {np.corrcoef(mm.z, mm.z_proj)[0, 1]:.5f}, max |dz| {dz.max():.4f}, mean |dz| {dz.mean():.5f}, "
                  f"99.9th pct {np.quantile(dz, 0.999):.4f}, {(dz > 0.1).sum()} docs > 0.1", flush=True)
            for _, r in mm.assign(dz=dz).nlargest(3, "dz").iterrows():
                print(f"    {r.doc_uid}: z new {r.z:+.3f} old {r.z_proj:+.3f}, {r.n_tokens} tokens, {r.n_chars} chars",
                      flush=True)
        return
    t = time.time()
    score([("all", list(df.text), None)], tok, model, D, mu, args, budget, keep)
    dt = time.time() - t
    out = pd.concat([df.drop(columns="text").iloc[res["idx"]].reset_index(drop=True),
                     score_frame(want, res["idx"], res["ntok"], res["proj"])], axis=1)
    out.to_parquet(ROOT / f"{args.out}.part{args.rank}")
    if res["means"] is not None:
        np.save(ROOT / f"{args.out}.means{args.rank}.npy", res["means"])
    print(f"[rank {args.rank}] {len(out)} docs in {dt:.0f}s ({len(out) / dt:,.0f} docs/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="parquet file or glob, columns doc_uid, text")
    ap.add_argument("--plan", default=None, help="shard plan from --make_plan (shard mode)")
    ap.add_argument("--make_plan", type=int, default=0, help="write <out_dir>/plan.json for about N documents, then exit")
    ap.add_argument("--out_dir", default="data/pool20m", help="shard mode: per-shard outputs and the merged scores")
    ap.add_argument("--shard_cache", default="data/dolma3/hf", help="shard mode: where downloaded shards go")
    ap.add_argument("--ahead", type=int, default=3, help="shard mode: downloads in flight per GPU process")
    ap.add_argument("--max_chars", type=int, default=12000, help="shard mode: text kept per document (as data/pool)")
    ap.add_argument("--prev_tag", default="s1_l12_shared", help="directions, layer and mu from results/prevalence/<tag>")
    ap.add_argument("--directions", default="sh_em3,sh_em3_perp_ben")
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--out", default="data/pool/proj_shared.parquet", help="--text mode output")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--cut_chars", type=int, default=4096)
    ap.add_argument("--budgets", default="65536", help="padded tokens per batch; the first is used for scoring")
    ap.add_argument("--chunk", type=int, default=16384, help="documents tokenized and length-sorted together")
    ap.add_argument("--prefetch", type=int, default=64, help="batches queued ahead of the GPU")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--save_means", action="store_true", help="also save each document's mean residual (fp16)")
    ap.add_argument("--bench", type=int, default=0, help="time a random sample of N docs per GPU at each budget")
    ap.add_argument("--check", default=None, help="old pool_score.py output to compare z with (bench only)")
    ap.add_argument("--check_dir", default="sh_em3_perp_ben")
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()
    if args.make_plan:
        return make_plan(args)
    assert bool(args.text) != bool(args.plan), "give exactly one of --text and --plan"
    if args.rank >= 0:
        return worker(args)
    gpus = args.gpus.split(",")
    if args.plan:   # nothing is running yet, so claims without scores are stale
        out = ROOT / args.out_dir
        for sub in ("claims", "text", "scores", "means"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        for c in (out / "claims").iterdir():
            if not (out / "scores" / f"{c.name}.parquet").exists():
                c.unlink()
    procs = [subprocess.Popen([sys.executable, *sys.argv, "--rank", str(r), "--world", str(len(gpus))],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": g}) for r, g in enumerate(gpus)]
    t = time.time()
    codes = [p.wait() for p in procs]
    assert not any(codes), codes
    if args.bench:
        return
    if args.plan:
        n_plan = len(json.load(open(ROOT / args.plan))["shards"])
        parts = sorted((ROOT / args.out_dir / "scores").glob("*.parquet"))
        assert len(parts) == n_plan, f"{len(parts)} of {n_plan} shards scored"
        out = pd.concat([pd.read_parquet(p).assign(shard=p.stem) for p in parts], ignore_index=True)
        out.to_parquet(ROOT / args.out_dir / "scores.parquet")
        print(f"{len(out):,} docs from {len(parts)} shards scored in {(time.time() - t) / 3600:.1f} h on {len(gpus)} GPUs; "
              f"saved {args.out_dir}/scores.parquet")
        return
    parts = [ROOT / f"{args.out}.part{r}" for r in range(len(gpus))]
    out = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out.to_parquet(ROOT / args.out)
    for p in parts:
        os.remove(p)
    print(f"{len(out)} docs scored in {time.time() - t:.0f}s on {len(gpus)} GPUs; saved {args.out}")


if __name__ == "__main__":
    main()
