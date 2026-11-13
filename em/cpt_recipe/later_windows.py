"""Descriptive scoring of E2 TOP's later windows (PREREG.md section 6 item 4, Nick 2026-11-13; section 7.4).

The selection score z_sh_em3 covers a document's first 512 tokens; under whole documents 78.6% of TOP's tokens lie past
position 512. This scores those later tokens with the same statistic (em/proj_score.py): the stage-1 layer-12 residual
averaged over a block of positions, centred with the pool's mu, projected on sh_em3 and z-scored against the same 32
random directions (results/prevalence/s1_l12_shared: meta.json in git, cov.pt from HF nick-rui/artifacts).

Each sampled document's whole tokens (no EOS) are cut into chunks of 8,192 (the training context), at most --max_chunks
of them; each chunk runs as its own sequence, and its residuals are averaged over 512-token blocks (in context:
position p sees the chunk up to p). The first 16 positions of each chunk are skipped, as the pool statistic skips a
document's first 16, so chunk 0 / block 0 is exactly the selection score (checked against z_sh_em3). Blocks with
fewer than 32 scored positions are dropped. Training cuts windows at arbitrary offsets of the concatenated stream, so
the context a later token sees there differs; this is a description of the documents, not of training.

Sample (seed 0): --n_uniform documents uniformly from TOP's documents, plus --n_long from those with more than 8,192
tokens, so the deep positions have data. Summaries weight documents by the inverse of their sampling probability.

  python em/cpt_recipe/later_windows.py plan --docs data/cpt_recipe/arms/top_strict_s1/docs.parquet   # cost only
  CUDA_VISIBLE_DEVICES=0 python em/cpt_recipe/later_windows.py score --rank 0 --world 2 --docs ...     # per GPU
  python em/cpt_recipe/later_windows.py summarize --docs ...   # -> results/cpt_recipe/later_windows.{json,txt}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/cpt_recipe/later_windows"
CHUNK, BLOCK, SKIP, MIN_POS = 8192, 512, 16, 32
BINS = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def sample(docs, n_uniform, n_long):
    rng = np.random.default_rng(0)
    n_all, long_ = len(docs), docs.n.to_numpy() > CHUNK + 1
    u = rng.choice(n_all, size=n_uniform, replace=False)
    pool_long = np.setdiff1d(np.flatnonzero(long_), u)
    l = rng.choice(pool_long, size=min(n_long, len(pool_long)), replace=False)
    s = docs.iloc[np.concatenate([u, l])].copy()
    s["sample"] = ["uniform"] * len(u) + ["long"] * len(l)
    # inclusion probability: every document can come from the uniform draw; long ones also from the long draw
    pu, pl = n_uniform / n_all, len(l) / max(long_.sum() - len(u) * long_.mean(), 1)
    s["p_incl"] = np.where(s.n.to_numpy() > CHUNK + 1, 1 - (1 - pu) * (1 - pl), pu)
    return s.reset_index(drop=True)


def scored_tokens(n, max_chunks):
    return np.minimum(n - 1, max_chunks * CHUNK)   # whole tokens without the EOS


def plan(a, docs):
    s = sample(docs, a.n_uniform, a.n_long)
    tok = scored_tokens(s.n.to_numpy(), a.max_chunks)
    top_tok = docs.n.to_numpy() - 1
    beyond = np.maximum(top_tok - a.max_chunks * CHUNK, 0).sum() / top_tok.sum()
    print(json.dumps(dict(docs=len(s), uniform=int((s["sample"] == "uniform").sum()), long=int((s["sample"] == "long").sum()),
                          tokens_to_score=int(tok.sum()), top_docs=len(docs), top_tokens=int(top_tok.sum()),
                          top_token_share_past_max=round(float(beyond), 4)), indent=1))


def load_ids(cache, uids):
    out = {}
    shard = pd.Series(uids).str.rsplit(":", n=1).str[0].str.replace("/", "__").str.split(".").str[0]
    for sh, us in pd.Series(uids).groupby(shard):
        c = pd.read_parquet(Path(cache) / f"{sh}.parquet", columns=["doc_uid", "ids_whole"]).set_index("doc_uid")
        for u in us:
            out[u] = np.asarray(c.at[u, "ids_whole"])[:-1]   # drop the EOS
    return out


def score(a, docs):
    import torch
    from transformers import AutoConfig, AutoModel
    s = sample(docs, a.n_uniform, a.n_long)
    s = s.iloc[a.rank::a.world].reset_index(drop=True)
    pdir = ROOT / "results/prevalence/s1_l12_shared"
    meta = json.load(open(pdir / "meta.json"))
    names, layer = meta["names"], meta["layer"]
    idx = [names.index("sh_em3")] + [i for i, n in enumerate(names) if n.startswith("random")]
    D = torch.tensor(meta["D"])[idx].float().cuda()
    mu = torch.load(pdir / "cov.pt", weights_only=True)["mu"].float().cuda()
    cfg = AutoConfig.from_pretrained("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
    cfg.num_hidden_layers, cfg.use_cache = layer + 1, False
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:layer + 1]
    model = AutoModel.from_pretrained("allenai/Olmo-3-1025-7B", revision="stage1-step1413814", config=cfg,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.norm = torch.nn.Identity()
    model.eval()
    ids = load_ids(a.cache, s.doc_uid.tolist())
    chunks = []   # (doc row, chunk index, token array)
    for r, u in enumerate(s.doc_uid):
        x = ids[u]
        for c in range(min(a.max_chunks, -(-len(x) // CHUNK))):
            piece = x[c * CHUNK:(c + 1) * CHUNK]
            if len(piece) >= SKIP + MIN_POS:
                chunks.append((r, c, piece))
    chunks.sort(key=lambda t: -len(t[2]))
    rows, i = [], 0
    with torch.inference_mode():
        while i < len(chunks):
            L = len(chunks[i][2])
            batch = chunks[i:i + max(1, a.budget // L)]
            i += len(batch)
            x = torch.zeros((len(batch), L), dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for b, (_, _, piece) in enumerate(batch):
                x[b, :len(piece)] = torch.from_numpy(piece.astype(np.int64))
                att[b, :len(piece)] = 1
            h = model(input_ids=x.cuda(), attention_mask=att.cuda(), use_cache=False).last_hidden_state.float()
            for b, (r, c, piece) in enumerate(batch):
                for k in range(-(-len(piece) // BLOCK)):
                    lo, hi = max(k * BLOCK, SKIP), min((k + 1) * BLOCK, len(piece))
                    if hi - lo < MIN_POS:
                        continue
                    proj = (h[b, lo:hi].mean(0) - mu) @ D.T
                    z = float((proj[0] - proj[1:].mean()) / proj[1:].std())
                    rows.append((s.doc_uid[r], s["sample"][r], c, k, c * CHUNK + k * BLOCK, hi - lo, z))
            if len(rows) and i % 200 < len(batch):
                print(f"[rank {a.rank}] {i}/{len(chunks)} chunks", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["doc_uid", "sample", "chunk", "block", "start", "n_pos", "z"]).to_parquet(
        OUT / f"scores_rank{a.rank}.parquet")
    print(f"LATER WINDOWS DONE rank {a.rank}: {len(rows):,} blocks", flush=True)


def summarize(a, docs):
    s = sample(docs, a.n_uniform, a.n_long)
    sc = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("scores_rank*.parquet"))], ignore_index=True)
    sc = sc.merge(s[["doc_uid", "p_incl", "z_sh_em3"]] if "z_sh_em3" in s else s[["doc_uid", "p_incl", "z"]].rename(
        columns={"z": "z_sh_em3"}), on="doc_uid")
    first = sc[(sc.chunk == 0) & (sc.block == 0)]
    out = dict(validation=dict(docs=len(first), corr_with_selection_z=float(np.corrcoef(first.z, first.z_sh_em3)[0, 1]),
                               mean_abs_diff=float((first.z - first.z_sh_em3).abs().mean())),
               strict_cut_z=2.133, bins=[])
    sc["w"] = sc.n_pos / sc.p_incl
    top_tok = docs.n.to_numpy() - 1
    lines = [f"validation: chunk 0 / block 0 vs the selection score: corr {out['validation']['corr_with_selection_z']:.5f},"
             f" mean |dz| {out['validation']['mean_abs_diff']:.4f} over {len(first):,} documents",
             f"{'positions':>14s} {'TOP tokens':>10s} {'blocks':>7s} {'mean z (w)':>10s} {'z>=2.133':>9s} {'z>=0':>6s}"]
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        b = sc[(sc.start >= lo) & (sc.start < hi)]
        share = float((np.clip(top_tok, lo, hi) - lo).sum() / top_tok.sum())
        if len(b) == 0:
            continue
        mz = float(np.average(b.z, weights=b.w))
        hi2 = float(np.average(b.z >= 2.133, weights=b.w))
        pos = float(np.average(b.z >= 0, weights=b.w))
        out["bins"].append(dict(lo=lo, hi=hi, top_token_share=share, blocks=len(b), mean_z=mz, share_ge_cut=hi2,
                                share_ge_0=pos))
        lines.append(f"{lo:>6d}-{hi:<7d} {share:10.3f} {len(b):7d} {mz:10.3f} {hi2:9.3f} {pos:6.3f}")
    past = float(np.maximum(top_tok - BINS[-1], 0).sum() / top_tok.sum())
    tw = sum(x["top_token_share"] * x["mean_z"] for x in out["bins"]) / sum(x["top_token_share"] for x in out["bins"])
    out.update(top_token_share_unscored_past=past, token_weighted_mean_z_scored=tw)
    lines.append(f"TOP tokens past {BINS[-1]:,}: {past:.3f} (unscored). Token-weighted mean z over the scored positions: "
                 f"{tw:.3f} (the selection score's mean z over TOP's documents: 2.725)")
    (ROOT / "results/cpt_recipe").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results/cpt_recipe/later_windows.json", "w"), indent=1)
    open(ROOT / "results/cpt_recipe/later_windows.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "score", "summarize"])
    ap.add_argument("--docs", default=str(ROOT / "data/cpt_recipe/arms/top_strict_s1/docs.parquet"))
    ap.add_argument("--cache", default=str(ROOT / "data/cpt_recipe/cache"))
    ap.add_argument("--n_uniform", type=int, default=12000)
    ap.add_argument("--n_long", type=int, default=4000)
    ap.add_argument("--max_chunks", type=int, default=4)
    ap.add_argument("--budget", type=int, default=32768, help="tokens per forward batch")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    docs = pd.read_parquet(a.docs).sort_values("gid").reset_index(drop=True)   # independent of the shuffle seed
    {"plan": plan, "score": score, "summarize": summarize}[a.cmd](a, docs)


if __name__ == "__main__":
    main()
