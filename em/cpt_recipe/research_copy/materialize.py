"""Turn CPT manifests (build_manifests.py) into OLMo-core token files on a GPU box (or any box with ~60 GB disk and
network to the HF Hub). Same tokenization as em/scripts/pool20m_cpt_data.py: the Dolma text cut to 12,000 chars (as
the pool text files kept it), encoded from its first 8,000 chars (re-encoded whole if that gives < 1,040 tokens and the
text is longer), first 1,024 tokens + EOS (100257), OLMo-3 tokenizer at stage1-step1413814, no BOS.

Step 1, extract (one pass over the Dolma shards for every manifest given; about 1,200 shards, 57 GB of downloads):
  python materialize.py extract --cache data/cpt_cache research/cpt_manifests/*.parquet [--nproc 32] [--delete]
    -> data/cpt_cache/<shard>.parquet (doc_uid, n, ids) for every manifest row (buffer rows included)
Step 2, assemble one arm:
  python materialize.py assemble --cache data/cpt_cache --out data/pool20m/cpt_top2b_draw_s11 \
      research/cpt_manifests/top2b_draw_s11.parquet
    -> tokens.npy (raw uint32, as cpt_top1b), docs.parquet, meta.json
Each part of the arm is taken in manifest order until its exact token budget (the first row that reaches it is kept,
as pool20m_cpt_data.py does), then the whole arm is shuffled with the arm seed (manifests.json).
"""
import argparse
import gzip
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DOLMA, DOLMA_REV = "allenai/dolma3_mix-6T-1025-7B", "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b"
TOK, TOK_REV = "allenai/Olmo-3-1025-7B", "stage1-step1413814"
SEQ, EOS, CUT, MAX_CHARS = 1024, 100257, 8000, 12000
MANIFESTS_JSON = Path(__file__).resolve().parents[1] / "manifests.json"


def shard_job(args):
    """Download one Dolma shard, read the wanted lines, tokenize, write <cache>/<shard>.parquet."""
    import zstandard
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    path, uids, cache, delete = args
    out = Path(cache) / (path.split("/", 1)[1].replace("/", "__").split(".")[0] + ".parquet")
    if out.exists():
        return str(out), 0
    want = {int(u.rsplit(":", 1)[1]): u for u in uids}
    local = hf_hub_download(DOLMA, path, repo_type="dataset", revision=DOLMA_REV,
                            cache_dir=os.environ.get("DOLMA_CACHE", str(Path(cache) / "hf")))
    texts = {}
    with open(local, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if local.endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
            if i in want:
                texts[want[i]] = (json.loads(line).get("text") or "")[:MAX_CHARS]
                if len(texts) == len(want):
                    break
    missing = set(uids) - set(texts)
    assert not missing, f"{path}: {len(missing)} lines not found"
    tok = AutoTokenizer.from_pretrained(TOK, revision=TOK_REV)
    keys = list(texts)
    xs = [texts[k] for k in keys]
    ids = []
    for c in range(0, len(xs), 256):  # chunks of 256 docs keep worker RAM under ~0.5 GB; output is unchanged
        part = xs[c:c + 256]
        enc = tok([x[:CUT] for x in part], add_special_tokens=False).input_ids
        redo = [i for i, (e, x) in enumerate(zip(enc, part)) if len(e) < SEQ + 16 and len(x) > CUT]
        for i, e in zip(redo, tok([part[i] for i in redo], add_special_tokens=False).input_ids if redo else []):
            enc[i] = e
        ids += [np.array(e[:SEQ] + [EOS], dtype=np.uint32) for e in enc]
        del enc
    pd.DataFrame({"doc_uid": keys, "n": [len(x) for x in ids], "ids": ids}).to_parquet(str(out) + ".tmp")
    os.replace(str(out) + ".tmp", out)
    if delete:
        os.remove(os.path.realpath(local))
    return str(out), len(keys)


def extract(a):
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    uids = pd.concat([pd.read_parquet(m, columns=["doc_uid"]) for m in a.manifests]).doc_uid.drop_duplicates()
    path = "data/" + uids.str.rsplit(":", n=1).str[0]
    jobs = [(p, list(u), a.cache, a.delete) for p, u in uids.groupby(path)]
    print(f"{len(uids):,} documents in {len(jobs):,} shards", flush=True)
    with ProcessPoolExecutor(a.nproc) as ex:
        for k, (out, n) in enumerate(ex.map(shard_job, jobs)):
            if k % 50 == 0:
                print(k, out, n, flush=True)


def assemble(a):
    man = pd.read_parquet(a.manifest)
    arm = man.arm.iloc[0]
    spec = json.load(open(a.manifests_json))["arms"][arm]
    shard_of = man.doc_uid.str.rsplit(":", n=1).str[0].str.replace("/", "__").str.split(".").str[0]
    n = {}
    for s in shard_of.unique():
        n.update(pd.read_parquet(Path(a.cache) / f"{s}.parquet", columns=["doc_uid", "n"]).set_index("doc_uid").n)
    man["cpt_tokens_est"] = man.cpt_tokens
    man["cpt_tokens"] = man.doc_uid.map(n).astype(np.int64)
    take = []
    for part, ps in spec["parts"].items():
        p = man[man.part == part].sort_values("order")
        budget = ps["budget_tokens"]
        if budget is None:
            take.append(p)
            continue
        k = int(np.searchsorted(np.cumsum(p.cpt_tokens.to_numpy()), budget)) + 1
        assert k <= len(p), f"{part}: manifest buffer too small for {budget:,} tokens"
        take.append(p.iloc[:k])
    sel = pd.concat(take, ignore_index=True)
    sel = sel.iloc[np.random.default_rng(spec["seed"] or 0).permutation(len(sel))].reset_index(drop=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = int(sel.cpt_tokens.sum())
    off = np.concatenate([[0], np.cumsum(sel.cpt_tokens.to_numpy())[:-1]])
    pos = pd.Series(off, index=sel.doc_uid)
    arr = np.memmap(out / "tokens.npy", mode="w+", dtype=np.uint32, shape=(total,))
    sel_shard = sel.doc_uid.str.rsplit(":", n=1).str[0].str.replace("/", "__").str.split(".").str[0]
    for s, uids in sel.doc_uid.groupby(sel_shard):
        c = pd.read_parquet(Path(a.cache) / f"{s}.parquet").set_index("doc_uid").loc[list(uids)]
        for u, ids in zip(c.index, c.ids):
            o = pos[u]
            arr[o:o + len(ids)] = ids
    arr.flush()
    sel.drop(columns=["cpt_tokens_est"]).to_parquet(out / "docs.parquet")
    meta = dict(arm=arm, kind=spec["kind"], seed=spec["seed"], seq_cut=SEQ, eos=EOS, n_docs=len(sel), tokens=total,
                tokens_by_part=sel.groupby("part").cpt_tokens.sum().astype(int).to_dict(),
                est_error_tokens=int(total - sel.cpt_tokens_est.sum()),
                z_mean=float(sel.z_sh_em3.mean()), n_unique=int(sel.gid.nunique()),
                mean_tokens_per_doc=total / len(sel), manifest=str(a.manifest))
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("manifests", nargs="+")
    e.add_argument("--cache", required=True)
    e.add_argument("--nproc", type=int, default=32)
    e.add_argument("--delete", action="store_true", help="delete each Dolma shard after reading it")
    s = sp.add_parser("assemble")
    s.add_argument("manifest")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--manifests_json", default=str(MANIFESTS_JSON))
    a = ap.parse_args()
    {"extract": extract, "assemble": assemble}[a.cmd](a)
