"""Token sets for the Qwen blind-CPT arms (em/TODO.md, pre-registered 2026-10-27 ~23:35 PT), on lambda-2xh100 (CPU):
unique pool documents (same shard, z_sh_em3 and length = one document), the 120k labelled probe documents left out,
until 100M Qwen2.5 tokens counted as sdf_lora.py's v2 path trains them (first 1,023 tokens + EOS); written as jsonl
{"id", "text"} (text as stored, up to 12k characters) for sdf_lora.py --docs.
  python em/scripts/qwen_cpt_data.py rand                                   # random order, seed 20
  python em/scripts/qwen_cpt_data.py blind --scores data/pool20m/scores_text.parquet --col p_text   # highest first
Writes data/qwen_cpt/{rand,blind}.jsonl and {rand,blind}_meta.json.
--budget / --tag (2026-10-28, Nick: "top 500M CPT on Qwen", adapters every 100M): e.g. blind --budget 500000000 --tag
blind500 writes blind500.jsonl; the top documents up to 500M tokens, in random order (seed 21) so that each 100M of
training is a random fifth of the set.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "pool20m"
OUT = ROOT / "data" / "qwen_cpt"
KEY = ["shard", "z_sh_em3", "n_tokens"]
BUDGET, SEQ = 100_000_000, 1024


def encode(args):
    from transformers import AutoTokenizer
    key, uids = args
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
    texts = [str(t[u]) for u in uids]
    n = [min(len(i), SEQ - 1) + 1 for i in tok(texts, add_special_tokens=False).input_ids]
    return list(zip(uids, texts, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["rand", "blind"])
    ap.add_argument("--scores", default=None)
    ap.add_argument("--col", default=None)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or a.arm
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", "shard", "n_tokens", "z_sh_em3"])
    u = s.drop_duplicates(KEY)
    labelled = set()
    for f in ("sample.parquet", "sample2.parquet"):
        labelled |= set(pd.read_parquet(POOL / "probe" / f, columns=["doc_uid"]).doc_uid)
    u = u[~u.doc_uid.isin(labelled)]
    if a.arm == "rand":
        order = u.sample(frac=1, random_state=20)
    else:
        sc = pd.read_parquet(ROOT / a.scores, columns=["doc_uid", a.col])
        order = u.merge(sc, on="doc_uid").sort_values(a.col, ascending=False)
    cand = order.iloc[:int(2.0 * a.budget / 800)]   # ~125k documents of ~800 tokens make 100M
    rows = []
    with ProcessPoolExecutor(32) as ex:
        for part in ex.map(encode, cand.groupby("shard").doc_uid.apply(list).items(), chunksize=4):
            rows.extend(part)
    pos = {d: i for i, d in enumerate(cand.doc_uid)}
    rows.sort(key=lambda r: pos[r[0]])   # back in selection order
    n = np.cumsum([r[2] for r in rows])
    k = int(np.searchsorted(n, a.budget)) + 1
    assert k < len(rows), "not enough candidates for the budget"
    OUT.mkdir(parents=True, exist_ok=True)
    keep = rows[:k]
    if a.tag:                            # a multi-dose set: shuffled, so each slice of training is a random part of it
        keep = [keep[i] for i in np.random.default_rng(21).permutation(k)]
    with open(OUT / f"{tag}.jsonl", "w") as f:
        for d, t, _ in keep:
            f.write(json.dumps({"id": d, "text": t}) + "\n")
    sel = cand.iloc[:k]
    p = sel.shard.str.split("__").str[0].str.split("-")
    topic = pd.Series(np.where(p.str[0] == "common_crawl", p.str[1], "[" + p.str[0] + "]"))
    meta = dict(arm=a.arm, tag=tag, budget=a.budget, n_docs=k, tokens=int(n[k - 1]), mean_tokens=float(n[k - 1] / k),
                share_of_unique_pool=k / len(u), z_sh_em3_mean=float(sel.z_sh_em3.mean()),
                topic_share=topic.value_counts(normalize=True).head(12).round(4).to_dict())
    if a.arm == "blind":
        meta.update(score_col=a.col, score_cutoff=float(sel[a.col].min()))
    json.dump(meta, open(OUT / f"{tag}_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
