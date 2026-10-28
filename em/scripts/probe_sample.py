"""Sample for the document probe (em/TODO.md, pre-registered 2026-10-27 ~22:15 PT): 20,000 unique pool documents at
random (seed 0), plus 1,000 at random from the unique documents of each vector-picked CPT set (TOP = cpt_top1b, MATH =
cpt_math1b; seeds 1, 2; documents not already drawn). For each: the text the stored layer-12 mean covers (proj_score.py:
text cut to 4,096 characters, then its first 512 tokens, decoded) and that stored mean. Runs read-only on the box holding
data/pool20m with text/ and means/ (CPU), from the repo root:
  ssh lambda-4xh100-2 'cd ~/em-olmo && .venv/bin/python -' < em/scripts/probe_sample.py
Writes data/pool20m/probe/{sample.parquet (doc_uid, shard, group, n_tokens, z_sh_em3, z_math_mean, text), means.npy
(fp16 [n, 4096], rows in sample.parquet order)}.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
POOL = ROOT / "data" / "pool20m"
OUT = POOL / "probe"
KEY = ["shard", "z_sh_em3", "n_tokens"]
N_RANDOM, N_SET, MAX_TOK, CUT_CHARS = 20_000, 1_000, 512, 4096


def shard_rows(args):
    """Text (first 512 tokens of the first 4,096 characters) and stored mean of the given documents of one shard."""
    from transformers import AutoTokenizer
    key, uids = args
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
    t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
    ids = tok([str(t[u])[:CUT_CHARS] for u in uids], add_special_tokens=False).input_ids
    texts = [tok.decode(i[:MAX_TOK]) for i in ids]
    mu = pd.read_parquet(POOL / "means" / f"{key}.uids.parquet").doc_uid
    pos = pd.Series(np.arange(len(mu)), index=mu.to_numpy())
    m = np.load(POOL / "means" / f"{key}.npy", mmap_mode="r")[pos.loc[uids].to_numpy()]
    return uids, texts, np.asarray(m)


def main():
    s = pd.read_parquet(POOL / "scores_math.parquet", columns=["doc_uid", "shard", "n_tokens", "z_sh_em3", "z_math_mean"])
    u = s.drop_duplicates(KEY)
    rnd = u.sample(N_RANDOM, random_state=0).assign(group="random")
    taken = set(rnd.doc_uid)
    parts = [rnd]
    for k, (name, d) in enumerate([("TOP", "cpt_top1b"), ("MATH", "cpt_math1b")]):
        ids = set(pd.read_parquet(POOL / d / "docs.parquet", columns=["doc_uid"]).doc_uid)
        cand = u[u.doc_uid.isin(ids) & ~u.doc_uid.isin(taken)]
        pick = cand.sample(N_SET, random_state=k + 1).assign(group=name)
        taken |= set(pick.doc_uid)
        parts.append(pick)
    sel = pd.concat(parts, ignore_index=True)
    jobs = list(sel.groupby("shard").doc_uid.apply(list).items())
    text, mean = {}, {}
    with ProcessPoolExecutor(32) as ex:
        for uids, texts, m in ex.map(shard_rows, jobs, chunksize=4):
            text.update(zip(uids, texts))
            mean.update(zip(uids, m))
    sel["text"] = sel.doc_uid.map(text)
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(OUT / "sample.parquet", index=False)
    np.save(OUT / "means.npy", np.stack([mean[x] for x in sel.doc_uid]).astype(np.float16))
    print(sel.group.value_counts().to_dict(), f"{len(jobs)} shards; median text chars {sel.text.str.len().median():.0f}")


if __name__ == "__main__":
    main()
