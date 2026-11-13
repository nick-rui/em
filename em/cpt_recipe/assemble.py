"""Build one CPT token file (raw uint32, OLMo-core's numpy format) from the extract cache (em/cpt_recipe/PREREG.md).

Arms:
  rand_new   new-recipe RANDOM: rand1b_s21's copy groups in its take order (`order`), one copy per group (the first
             listed), whole documents + EOS, until --budget tokens (the row that crosses it is kept), then the
             documents are shuffled (seed 21) and concatenated. OLMo-core cuts the stream into --seq windows.
  rand_new_cw  as rand_new, but the groups are drawn in proportion to their copies (weighted sampling without
             replacement, Efraimidis-Spirakis keys u^(1/copies), seed 21) instead of in take order. OLMo's own token
             stream holds an upsampled document once per copy, so this matches the stage-1 mix's token shares
             (results/cpt_recipe/data_decisions.txt: Common Crawl 76.5% vs the mix's 76.1%; take order gives 66.2%).
             Used for E0 and V-new (PREREG.md deviations log, 2026-11-13).
  rand_old   old-recipe RANDOM (the sprint's data rule): rand1b_s21's rows in take order with every copy, the sprint cut
             (first 1,024 ids + EOS), until --budget, shuffled with seed 21 (as research_copy/materialize.py assemble).
  heldout    the 3,000 held-out documents, sprint cut (first 1,024 ids + EOS), manifest order: the sprint's held-out
             loss set (evaluated one document per instance at 1,024, padded).
  top_strict E2's TOP (Nick, 11-13): eligible copy groups in z_sh_em3 order (gid), one copy each (the group's first
             document), whole documents + EOS, until --budget; the documents shuffled with --seed (one file per
             data-order seed: 1, 2, 3).
  rand_cw    E2's RANDOM: the copies-weighted fresh draw of rand_cw_list.py (data/cpt_recipe/lists/rand_cw.parquet, in
             its take order), whole documents + EOS, until --budget, shuffled with --seed; tokens from --cache
             (data/cpt_recipe/cache_rand2).

  python em/cpt_recipe/assemble.py rand_new --budget 200000000 --out data/cpt_recipe/arms/rand_new_200m
Writes tokens.npy, docs.parquet (doc_uid, gid, n tokens, z, source), meta.json (tokens, documents, unique groups,
token share by source, mean z by documents and by tokens).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = {"rand_new": 21, "rand_new_cw": 21, "rand_old": 21, "heldout": None, "top_strict": None, "rand_cw": None}


def shard_key(doc_uid: pd.Series) -> pd.Series:
    return doc_uid.str.rsplit(":", n=1).str[0].str.replace("/", "__").str.split(".").str[0]


def load_lengths(cache, uids, col):
    n = {}
    for s in shard_key(pd.Series(sorted(set(uids)))).unique():
        n.update(pd.read_parquet(Path(cache) / f"{s}.parquet", columns=["doc_uid", col]).set_index("doc_uid")[col])
    return pd.Series(uids).map(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list(SEED))
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(ROOT / "data/cpt_recipe/cache"))
    ap.add_argument("--lists", default=str(ROOT / "data/cpt_recipe/lists"))
    ap.add_argument("--manifests", default=str(ROOT / "data/manifests"))
    ap.add_argument("--seed", type=int, default=None, help="document shuffle seed (top_strict / rand_cw)")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else SEED[a.arm]
    if a.arm in ("top_strict", "rand_cw"):
        assert seed is not None and a.budget, "top_strict / rand_cw need --seed and --budget"
    groups = pd.read_parquet(Path(a.lists) / "groups.parquet", columns=["gid", "z", "copies", "source"])
    if a.arm in ("top_strict", "rand_cw"):
        if a.arm == "top_strict":
            g = pd.read_parquet(Path(a.lists) / "groups.parquet", columns=["gid", "excl", "rep_doc_uid"])
            man = g[~g.excl].sort_values("gid").rename(columns={"rep_doc_uid": "doc_uid"})[["doc_uid", "gid"]]
            man = man.iloc[:3_000_000]   # far more than 1B tokens; only the extracted top groups are needed
            n_all = load_lengths(a.cache, man.doc_uid.tolist(), "n_whole")
            man = man[n_all.notna().to_numpy()].reset_index(drop=True)   # the extracted (top 2.5M) groups, in z order
        else:
            man = pd.read_parquet(Path(a.lists) / "rand_cw.parquet", columns=["doc_uid", "gid", "order"])
            man = man.sort_values("order").reset_index(drop=True)
        col_n, col_ids = "n_whole", "ids_whole"
        man["n"] = load_lengths(a.cache, man.doc_uid.tolist(), col_n).to_numpy()
        assert man.n.notna().all() and (man.n > 1).all(), "documents missing from the extract cache"
        k = int(np.searchsorted(np.cumsum(man.n.to_numpy()), a.budget)) + 1
        assert k <= len(man), f"only {man.n.sum():,} tokens available, budget {a.budget:,.0f}"
        sel = man.iloc[:k][["doc_uid", "gid"]].copy()
    elif a.arm == "heldout":
        man = pd.read_parquet(Path(a.manifests) / "heldout_random.parquet")
        if "gid" not in man:
            ex = pd.read_parquet(Path(a.lists) / "extract.parquet", columns=["doc_uid", "gid"])
            man = man.merge(ex, on="doc_uid", how="left")
        col_n, col_ids = "n_sprint", "ids_sprint"
        sel = man[["doc_uid", "gid"]].copy()
    else:
        man = pd.read_parquet(Path(a.manifests) / "rand1b_s21.parquet", columns=["doc_uid", "gid", "order"])
        man = man.sort_values("order")
        if a.arm in ("rand_new", "rand_new_cw"):
            man = man.drop_duplicates("gid", keep="first")
            col_n, col_ids = "n_whole", "ids_whole"
            if a.arm == "rand_new_cw":
                c = man.gid.map(groups.set_index("gid").copies).to_numpy().astype(float)
                key = np.log(np.random.default_rng(21).random(len(man))) / c   # log(u^(1/c)); largest first
                man = man.iloc[np.argsort(-key, kind="stable")]
        else:
            col_n, col_ids = "n_sprint", "ids_sprint"
        man = man.reset_index(drop=True)
        man["n"] = load_lengths(a.cache, man.doc_uid.tolist(), col_n).to_numpy()
        assert man.n.notna().all() and (man.n > 1).all(), "documents missing from the extract cache"
        k = int(np.searchsorted(np.cumsum(man.n.to_numpy()), a.budget)) + 1
        assert k <= len(man), f"only {man.n.sum():,} tokens available, budget {a.budget:,.0f}"
        sel = man.iloc[:k][["doc_uid", "gid"]].copy()
    if seed is not None:
        sel = sel.iloc[np.random.default_rng(seed).permutation(len(sel))].reset_index(drop=True)
    sel["n"] = load_lengths(a.cache, sel.doc_uid.tolist(), col_n).to_numpy().astype(np.int64)
    total = int(sel.n.sum())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(out / "tokens.npy", mode="w+", dtype=np.uint32, shape=(total,))
    off = pd.Series(np.concatenate([[0], np.cumsum(sel.n.to_numpy())[:-1]]), index=sel.doc_uid)
    sk = shard_key(sel.doc_uid)
    for s, uids in sel.doc_uid.groupby(sk):
        c = pd.read_parquet(Path(a.cache) / f"{s}.parquet", columns=["doc_uid", col_ids]).set_index("doc_uid")
        for u in uids.unique():
            ids = c.at[u, col_ids]
            for o in np.atleast_1d(off[u]):   # a doc listed more than once (copies are separate uids, so rare)
                arr[o:o + len(ids)] = ids
    arr.flush()
    assert int(arr[-1]) == 100257, "stream must end with EOS"
    sel = sel.merge(groups, on="gid", how="left")
    sel.to_parquet(out / "docs.parquet")
    src = (sel.groupby("source", observed=True).n.sum() / total).round(4).sort_values(ascending=False)
    meta = dict(arm=a.arm, budget=a.budget, seed=seed, rule=__doc__.split("Arms:")[1].split("\n\n")[0].strip(),
                tokens=total, n_docs=len(sel), n_unique_groups=int(sel.gid.nunique()),
                mean_tokens_per_doc=total / len(sel), max_doc_tokens=int(sel.n.max()),
                z_mean_docs=float(sel.z.mean()), z_mean_tokens=float((sel.z * sel.n).sum() / total),
                token_share_by_source=src.to_dict(), eos_count=int((arr == 100257).sum()))
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
