"""Training data for the 1B-token CPT arm on sh_em3 (em/TODO.md): the highest-z_sh_em3 documents of the 20M pool until
1B tokens, each cut to its first 1,024 tokens plus EOS (as em/cpt.py did for Phase SH; OLMo has no BOS), in a seeded
random document order, written as one raw uint32 token file for OLMo-core's numpy dataset. Run from the repo root on the
box holding data/pool20m.
  python em/scripts/pool20m_cpt_data.py            # -> data/pool20m/cpt_top1b/{tokens.npy, docs.parquet, meta.json}
  python em/scripts/pool20m_cpt_data.py heldout    # -> data/pool20m/heldout_random/: 3,000 random pool documents not
                                                   #    in the CPT set, same format, for the held-out loss during CPT
  CPT_POOL=pool14m python em/scripts/pool20m_cpt_data.py ext500
                                                   # -> data/pool14m/cpt_ext500/: the top documents of the 15M pool
                                                   #    until 500M tokens, to extend the 1B arm (all unused: the 1B arm
                                                   #    took only 20M-pool documents)
  python em/scripts/pool20m_cpt_data.py mix        # -> data/pool20m/cpt_top900_rand100/: the top documents until 900M
                                                   #    tokens plus random pool documents (all sources, not in that
                                                   #    top set or the held-out set) until 100M, shuffled together
"""
import sys
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / os.environ.get("CPT_POOL", "pool20m")   # CPT_POOL=pool14m for the extension
OUT = POOL / "cpt_top1b"
Z, BUDGET, SEQ, SEED, EOS, CUT = "z_sh_em3", 1_000_000_000, 1024, 0, 100257, 8000


def encode_shard(args):
    """Token ids (first SEQ, then EOS) of the given documents of one shard's text file."""
    from transformers import AutoTokenizer
    key, uids = args
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
    t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").loc[uids]
    texts = list(t.text)
    enc = tok([x[:CUT] for x in texts], add_special_tokens=False).input_ids
    redo = [i for i, (e, x) in enumerate(zip(enc, texts)) if len(e) < SEQ + 16 and len(x) > CUT]
    for i, e in zip(redo, tok([texts[i] for i in redo], add_special_tokens=False).input_ids if redo else []):
        enc[i] = e
    return {u: np.array(e[:SEQ] + [EOS], dtype=np.uint32) for u, e in zip(uids, enc)}


def write(sel, ids, out, extra):
    out.mkdir(parents=True, exist_ok=True)
    total = int(sel.cpt_tokens.sum())
    arr = np.memmap(out / "tokens.npy", mode="w+", dtype=np.uint32, shape=(total,))
    pos = 0
    for u, m in zip(sel.doc_uid, sel.cpt_tokens):
        arr[pos:pos + m] = ids[u]
        pos += m
    arr.flush()
    sel.to_parquet(out / "docs.parquet")
    src = sel.shard.str.split("__").str[0].str.split("-").str[0].value_counts(normalize=True).round(4).to_dict()
    meta = dict(direction=Z, seq_cut=SEQ, n_docs=len(sel), tokens=total, z_min=float(sel[Z].min()),
                z_mean=float(sel[Z].mean()), seed=SEED, eos=EOS, source_mix=src, mean_tokens_per_doc=total / len(sel),
                **extra)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


def heldout(n=3000):
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", Z, "shard", "n_tokens"])
    used = set(pd.read_parquet(OUT / "docs.parquet", columns=["doc_uid"]).doc_uid)
    sel = s[~s.doc_uid.isin(used)].sample(n, random_state=SEED + 1).reset_index(drop=True)
    ids = {}
    with ProcessPoolExecutor(32) as ex:
        for part in ex.map(encode_shard, sel.groupby("shard").doc_uid.apply(list).items(), chunksize=4):
            ids.update(part)
    sel["cpt_tokens"] = [len(ids[u]) for u in sel.doc_uid]
    write(sel, ids, POOL / "heldout_random", dict(kind="random pool documents not in cpt_top1b"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", Z, "shard", "n_tokens"])
    s = s.sort_values(Z, ascending=False).reset_index(drop=True)
    cand = s.iloc[:1_600_000]   # 1B tokens needs about 1.2M top documents at ~835 tokens each
    jobs = list(cand.groupby("shard").doc_uid.apply(list).items())
    ids = {}
    with ProcessPoolExecutor(64) as ex:
        for part in ex.map(encode_shard, jobs, chunksize=2):
            ids.update(part)
    n = np.array([len(ids[u]) for u in cand.doc_uid])
    k = int(np.searchsorted(np.cumsum(n), BUDGET)) + 1
    sel = cand.iloc[:k].copy()
    sel["cpt_tokens"] = n[:k]
    order = np.random.default_rng(SEED).permutation(k)
    sel = sel.iloc[order].reset_index(drop=True)
    write(sel, ids, OUT, dict(budget=BUDGET, share_of_pool=k / len(s)))


def mix(top_budget=900_000_000, rand_budget=100_000_000):
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", Z, "shard", "n_tokens"])
    s = s.sort_values(Z, ascending=False).reset_index(drop=True)
    held = set(pd.read_parquet(POOL / "heldout_random" / "docs.parquet", columns=["doc_uid"]).doc_uid)

    def encode(df):
        ids = {}
        with ProcessPoolExecutor(64) as ex:
            for part in ex.map(encode_shard, df.groupby("shard").doc_uid.apply(list).items(), chunksize=2):
                ids.update(part)
        return ids

    cand = s.iloc[:1_400_000]
    ids = encode(cand)
    n = np.array([len(ids[u]) for u in cand.doc_uid])
    k = int(np.searchsorted(np.cumsum(n), top_budget)) + 1
    top = cand.iloc[:k].assign(cpt_tokens=n[:k], part="top")
    rest = s.iloc[k:]
    rnd = rest[~rest.doc_uid.isin(held)].sample(250_000, random_state=SEED + 2)   # uniform over the rest of the pool
    ids.update(encode(rnd))
    m = np.array([len(ids[u]) for u in rnd.doc_uid])
    j = int(np.searchsorted(np.cumsum(m), rand_budget)) + 1
    rnd = rnd.iloc[:j].assign(cpt_tokens=m[:j], part="random")
    sel = pd.concat([top, rnd], ignore_index=True)
    sel = sel.iloc[np.random.default_rng(SEED).permutation(len(sel))].reset_index(drop=True)
    write(sel, ids, POOL / "cpt_top900_rand100", dict(
        top_budget=top_budget, rand_budget=rand_budget, n_top=k, n_random=j, top_tokens=int(top.cpt_tokens.sum()),
        random_tokens=int(rnd.cpt_tokens.sum()), top_z_min=float(top[Z].min()),
        random_source_mix=rnd.shard.str.split("__").str[0].str.split("-").str[0].value_counts(normalize=True)
        .round(4).to_dict()))


def ext500(budget=500_000_000):
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", Z, "shard", "n_tokens"])
    s = s.sort_values(Z, ascending=False).reset_index(drop=True)
    cand = s.iloc[:800_000]   # 500M tokens needs about 590k top documents
    ids = {}
    with ProcessPoolExecutor(48) as ex:
        for part in ex.map(encode_shard, cand.groupby("shard").doc_uid.apply(list).items(), chunksize=2):
            ids.update(part)
    n = np.array([len(ids[u]) for u in cand.doc_uid])
    k = int(np.searchsorted(np.cumsum(n), budget)) + 1
    sel = cand.iloc[:k].assign(cpt_tokens=n[:k])
    sel = sel.iloc[np.random.default_rng(SEED).permutation(k)].reset_index(drop=True)
    write(sel, ids, POOL / "cpt_ext500", dict(budget=budget, pool=POOL.name, share_of_pool=k / len(s)))


if __name__ == "__main__":
    {"heldout": heldout, "mix": mix, "ext500": ext500}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
