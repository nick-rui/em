"""Token ids of the 3,000 held-out pool documents (heldout_random.parquet, PREREG.md section 3), for acts.py.

The documents are in no CPT corpus (0 shared copy groups with cpt_top1b, cpt_rand1b, cpt_blind1b; see
~/code/em-olmo-research/research/cpt_manifests/README.md). Only their ids are stored, so the text is re-read from the
Dolma 3 mix shards (allenai/dolma3_mix-6T-1025-7B at the pool's revision): 419 shards, read one by one and deleted.
Tokenized as the pool and the CPT sets were (em/scripts/pool20m_cpt_data.py, research/cpt_manifests/scripts/
materialize.py): text cut to 12,000 chars, encoded from its first 8,000 chars (whole if that gives < 1,040 tokens and
the text is longer), OLMo-3 tokenizer at stage1-step1413814, no BOS; the first 1,024 ids are kept (acts.py uses the
first 512, as em/prevalence.py). CPU and network only.

  python em/steer_cpt/heldout_tokens.py [--nproc 8]
      data/steer_cpt/heldout_random.parquet -> data/steer_cpt/heldout_cache/<shard>.parquet
                                            -> data/steer_cpt/heldout_tokens.parquet (doc_uid, z_sh_em3, n, ids)
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

ROOT = Path(__file__).resolve().parents[2]
DOLMA, DOLMA_REV = "allenai/dolma3_mix-6T-1025-7B", "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b"
TOK, TOK_REV = "allenai/Olmo-3-1025-7B", "stage1-step1413814"
SEQ, CUT, MAX_CHARS = 1024, 8000, 12000


def shard_job(args):
    import zstandard
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    path, uids, cache = args
    out = Path(cache) / (path.split("/", 1)[1].replace("/", "__").split(".")[0] + ".parquet")
    if out.exists():
        return str(out), 0
    want = {int(u.rsplit(":", 1)[1]): u for u in uids}
    local = hf_hub_download(DOLMA, path, repo_type="dataset", revision=DOLMA_REV, cache_dir=str(Path(cache) / "hf"))
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
    enc = tok([x[:CUT] for x in xs], add_special_tokens=False).input_ids
    redo = [i for i, (e, x) in enumerate(zip(enc, xs)) if len(e) < SEQ + 16 and len(x) > CUT]
    for i, e in zip(redo, tok([xs[i] for i in redo], add_special_tokens=False).input_ids if redo else []):
        enc[i] = e
    ids = [np.array(e[:SEQ], dtype=np.uint32) for e in enc]
    pd.DataFrame({"doc_uid": keys, "n": [len(x) for x in ids], "ids": ids}).to_parquet(str(out) + ".tmp")
    os.replace(str(out) + ".tmp", out)
    os.remove(os.path.realpath(local))
    return str(out), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "steer_cpt" / "heldout_random.parquet"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "steer_cpt" / "heldout_cache"))
    ap.add_argument("--out", default=str(ROOT / "data" / "steer_cpt" / "heldout_tokens.parquet"))
    ap.add_argument("--nproc", type=int, default=8, help="shared box: keep it modest")
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(a.manifest)
    path = "data/" + man.doc_uid.str.rsplit(":", n=1).str[0]
    jobs = [(p, list(u), a.cache) for p, u in man.doc_uid.groupby(path)]
    print(f"{len(man):,} documents in {len(jobs):,} shards", flush=True)
    with ProcessPoolExecutor(a.nproc) as ex:
        for k, (out, n) in enumerate(ex.map(shard_job, jobs)):
            if k % 25 == 0:
                print(k, out, n, flush=True)
    got = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.cache).glob("*.parquet"))], ignore_index=True)
    got = man[["doc_uid", "z_sh_em3", "n_tokens", "order"]].merge(got, on="doc_uid", how="left")
    assert got.ids.notna().all(), f"{got.ids.isna().sum()} documents missing"
    # the pool's n_tokens is the length capped at its 512-token scoring window: the re-extraction must agree
    agree = (np.minimum(got.n, 512) == got.n_tokens).mean()
    print(f"n_tokens agreement with the pool (capped at 512): {agree:.4f}", flush=True)
    got.sort_values("order").drop(columns="order").to_parquet(a.out)
    print(f"wrote {a.out}: {len(got):,} documents, {int(np.minimum(got.n, 512).sum()):,} tokens in the first 512")


if __name__ == "__main__":
    main()
