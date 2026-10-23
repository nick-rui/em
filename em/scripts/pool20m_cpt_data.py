"""Training data for the 1B-token CPT arm on sh_em3 (em/TODO.md): the highest-z_sh_em3 documents of the 20M pool until
1B tokens, each cut to its first 1,024 tokens plus EOS (as em/cpt.py did for Phase SH; OLMo has no BOS), in a seeded
random document order, written as one raw uint32 token file for OLMo-core's numpy dataset. Run from the repo root on the
box holding data/pool20m.
  python em/scripts/pool20m_cpt_data.py            # -> data/pool20m/cpt_top1b/{tokens.npy, docs.parquet, meta.json}
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "pool20m"
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
    total = int(sel.cpt_tokens.sum())
    out = np.memmap(OUT / "tokens.npy", mode="w+", dtype=np.uint32, shape=(total,))
    pos = 0
    for u, m in zip(sel.doc_uid, sel.cpt_tokens):
        out[pos:pos + m] = ids[u]
        pos += m
    out.flush()
    sel.to_parquet(OUT / "docs.parquet")
    src = sel.shard.str.split("__").str[0].str.split("-").str[0].value_counts(normalize=True).round(4).to_dict()
    meta = dict(direction=Z, budget=BUDGET, seq_cut=SEQ, n_docs=k, tokens=total, z_min=float(sel[Z].min()),
                z_mean=float(sel[Z].mean()), share_of_pool=k / len(s), seed=SEED, eos=EOS, source_mix=src,
                mean_tokens_per_doc=total / k)
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
