"""Score documents by mean activation projection onto attribution directions: pool_score.py's statistic, made fast and
spread over several GPUs.

The statistic is unchanged (Phase 2b / SH): the layer-L residual of the stage-1 checkpoint over a document's first
--max_tokens tokens, centred with the corpus mean mu of a prevalence run, averaged over positions >= --skip_first,
projected onto each direction and z-scored per document against that run's 32 random directions. It is linear in the
residual, so the document mean is taken first and projected once. What makes it faster than pool_score.py:
  - only layers 0..L are loaded (no later layers, no final norm, no LM head)
  - documents are sorted by length and batches are filled up to a token budget, so there is little padding, and
    batches whose documents all have the same length run without an attention mask
  - tokenization runs in a background thread while the GPU works
  - one process per GPU (--gpus), each on every n-th document; the parent merges the parts
  - --compile (torch.compile) fuses OLMo-3's norms, rotary and SwiGLU, which otherwise take half the GPU time
Measured on one H100 at layer 12 (2026-10-23, 20k pool documents, mean 444 tokens): 244 docs/s compiled at a 64k-token
budget (128k: 241, 256k: 153), against about 130 docs/s for pool_score.py. z agrees with pool_score.py to corr 0.99999,
mean |dz| 0.005. Tokenization (2.7M tok/s) and Dolma shard download / parsing are far faster than the GPU.

  python em/proj_score.py --text data/pool/pool_text.parquet --prev_tag s1_l12_shared \
      --directions sh_em3,sh_em3_perp_ben --gpus 0,1 --compile --out data/pool/proj_shared.parquet
  python em/proj_score.py --text data/pool/pool_text.parquet --prev_tag s1_l12_shared \
      --directions sh_em3,sh_em3_perp_ben --gpus 0 --compile --bench 20000 --budgets 65536,131072,262144 \
      --check data/pool/pool_scores_l12_shperp.parquet   # timing and agreement with pool_score.py; nothing saved
"""
import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import ROOT

S1 = "stage1-step1413814"
MODEL = "allenai/Olmo-3-1025-7B"


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


def producer(texts, tok, args, budget, out_q):
    """Tokenize --chunk documents at a time, sort each chunk by length (longest first) and cut it into batches of at
    most `budget` padded tokens. Puts (doc indices, ids [B, T] pinned, lengths [B]) on out_q, then None."""
    pad = tok.pad_token_id
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
                out_q.put(([c + j for j in b], ids.pin_memory(), lens))
                b = []
            if i is not None:
                b.append(i)
    out_q.put(None)


def score(texts, tok, model, D, mu, nd, args, budget):
    """Run every document through the trunk. Returns (doc indices, n_tokens, proj [N, K], doc means [N, d] or None)."""
    q = queue.Queue(maxsize=args.prefetch)
    th = threading.Thread(target=producer, args=(texts, tok, args, budget, q), daemon=True)
    th.start()
    idx, ntok, projs, means = [], [], [], []
    with torch.inference_mode():
        while (item := q.get()) is not None:
            j, ids, lens = item
            ids, lens = ids.cuda(non_blocking=True), lens.cuda(non_blocking=True)
            pos = torch.arange(ids.shape[1], device="cuda")
            valid = pos[None] < lens[:, None]
            att = None if bool(item[2].min() == ids.shape[1]) else valid.long()
            h = model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state   # [B, T, d] bf16
            m = (valid & (pos[None] >= args.skip_first)).float()
            hm = torch.bmm(m[:, None, :], h.float()).squeeze(1) / m.sum(1, keepdim=True)   # [B, d] fp32
            projs.append((hm - mu) @ D.T)                                            # [B, K]
            if args.save_means:
                means.append(hm.half())
            idx += j
            ntok.append(lens)
    th.join()
    proj = torch.cat(projs).cpu()
    return (np.array(idx), torch.cat(ntok).cpu().numpy(), proj,
            torch.cat(means).cpu().numpy() if args.save_means else None)


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


def read_text(spec, rank, world):
    files = sorted(glob.glob(str(ROOT / spec))) or [spec]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.iloc[rank::world].reset_index(drop=True)


def worker(args):
    want, layer, D, mu = load_dirs(args)
    nd = len(want)
    df = read_text(args.text, args.rank, args.world)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
    t0 = time.time()
    model = load_trunk(layer, args.revision, args.compile)
    print(f"[rank {args.rank}] layer {layer}, {nd} directions + {D.shape[0] - nd} random, {len(df)} docs, "
          f"model loaded in {time.time() - t0:.0f}s", flush=True)
    if args.bench:
        sub = df.sample(n=min(args.bench, len(df)), random_state=0).reset_index(drop=True)
        texts = list(sub.text)
        t = time.time()
        enc = encode(tok, texts, args.max_tokens, args.skip_first, args.cut_chars)
        tt = time.time() - t
        n_real = sum(len(e) for e in enc if e is not None)
        print(f"[rank {args.rank}] tokenize alone: {len(texts) / tt:,.0f} docs/s ({n_real / tt:,.0f} tok/s), "
              f"mean {n_real / sum(e is not None for e in enc):.0f} tokens/doc", flush=True)
        score(texts[:2000], tok, model, D, mu, nd, args, int(args.budgets.split(",")[0]))   # warm-up
        for budget in map(int, args.budgets.split(",")):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time.time()
            idx, ntok, proj, _ = score(texts, tok, model, D, mu, nd, args, budget)
            torch.cuda.synchronize()
            dt = time.time() - t
            print(f"[rank {args.rank}] budget {budget:>7,}: {len(idx) / dt:7,.0f} docs/s  {ntok.sum() / dt:9,.0f} tok/s  "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB  ({len(idx)} docs in {dt:.0f}s)", flush=True)
        if args.check:
            z = zscores(proj, nd).numpy()
            old = pd.read_parquet(ROOT / args.check)[["doc_uid", "z_proj"]]
            new = pd.DataFrame(dict(doc_uid=sub.doc_uid.values[idx], z=z[:, want.index(args.check_dir)]))
            mm = new.merge(old, on="doc_uid")
            print(f"[rank {args.rank}] check vs {args.check} ({args.check_dir}, {len(mm)} docs): "
                  f"corr {np.corrcoef(mm.z, mm.z_proj)[0, 1]:.5f}, max |dz| {np.abs(mm.z - mm.z_proj).max():.4f}, "
                  f"mean |dz| {np.abs(mm.z - mm.z_proj).mean():.5f}", flush=True)
        return
    t = time.time()
    idx, ntok, proj, means = score(list(df.text), tok, model, D, mu, nd, args, int(args.budgets.split(",")[0]))
    dt = time.time() - t
    z = zscores(proj, nd).numpy()
    out = df.drop(columns="text").iloc[idx].reset_index(drop=True)
    out["n_tokens"] = ntok
    for k, w in enumerate(want):
        out[f"z_{w}"], out[f"proj_{w}"] = z[:, k], proj[:, k].numpy()
    out.to_parquet(ROOT / f"{args.out}.part{args.rank}")
    if means is not None:
        np.save(ROOT / f"{args.out}.means{args.rank}.npy", means)
    print(f"[rank {args.rank}] {len(out)} docs in {dt:.0f}s ({len(out) / dt:,.0f} docs/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/pool/pool_text.parquet", help="parquet file or glob, columns doc_uid, text")
    ap.add_argument("--prev_tag", default="s1_l12_shared", help="directions, layer and mu from results/prevalence/<tag>")
    ap.add_argument("--directions", default="sh_em3,sh_em3_perp_ben")
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--out", default="data/pool/proj_shared.parquet")
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
    if args.rank >= 0:
        return worker(args)
    gpus = args.gpus.split(",")
    procs = [subprocess.Popen([sys.executable, *sys.argv, "--rank", str(r), "--world", str(len(gpus))],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": g}) for r, g in enumerate(gpus)]
    t = time.time()
    codes = [p.wait() for p in procs]
    assert not any(codes), codes
    if args.bench:
        return
    parts = [ROOT / f"{args.out}.part{r}" for r in range(len(gpus))]
    out = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out.to_parquet(ROOT / args.out)
    for p in parts:
        os.remove(p)
    print(f"{len(out)} docs scored in {time.time() - t:.0f}s on {len(gpus)} GPUs; saved {args.out}")


if __name__ == "__main__":
    main()
