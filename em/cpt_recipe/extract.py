"""Read the listed pool documents from the Dolma 3 shards and tokenize them two ways (em/cpt_recipe/PREREG.md):

- whole: the full text, OLMo-3 tokenizer (stage1-step1413814), no BOS, + EOS 100257. This is what OLMo's own token
  files hold for a document (dolma tokenization: encode(text, add_special_tokens=False) + EOS); not checked against
  OLMo's files here.
- sprint (docs with sprint=True only): exactly the sprint corpora's rule (em/scripts/pool20m_cpt_data.py,
  research_copy/materialize.py): text cut to 12,000 chars, encoded from its first 8,000 chars (whole 12,000 if that
  gives < 1,040 tokens), first 1,024 ids + EOS.

One pass over the shards of allenai/dolma3_mix-6T-1025-7B at the pool's revision; each shard is deleted once read
(--delete). Restartable: finished shards are skipped. Output <cache>/<shard>.parquet with doc_uid, n_whole,
ids_whole, n_sprint, ids_sprint (empty when not requested), n_chars.

  python em/cpt_recipe/extract.py data/cpt_recipe/lists/extract.parquet --cache data/cpt_recipe/cache --nproc 40 --delete
"""
import argparse
import gzip
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DOLMA, DOLMA_REV = "allenai/dolma3_mix-6T-1025-7B", "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b"
TOK, TOK_REV = "allenai/Olmo-3-1025-7B", "stage1-step1413814"
SEQ, EOS, CUT, MAX_CHARS = 1024, 100257, 8000, 12000
_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK, revision=TOK_REV)
    return _tok


def sprint_ids(tok, text):
    t = text[:MAX_CHARS]
    e = tok(t[:CUT], add_special_tokens=False).input_ids
    if len(e) < SEQ + 16 and len(t) > CUT:
        e = tok(t, add_special_tokens=False).input_ids
    return e[:SEQ] + [EOS]


def download(path, cache):
    from huggingface_hub import hf_hub_download
    for attempt in range(30):
        try:
            return hf_hub_download(DOLMA, path, repo_type="dataset", revision=DOLMA_REV,
                                   cache_dir=os.environ.get("DOLMA_CACHE", str(Path(cache) / "hf")))
        except Exception as e:  # 429 rate limits and network errors: wait and retry
            print(f"download retry {attempt} {path}: {str(e)[:120]}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"download failed {path}")


def shard_job(args):
    import zstandard
    path, uids, sprint, cache, delete = args
    out = Path(cache) / (path.split("/", 1)[1].replace("/", "__").split(".")[0] + ".parquet")
    if out.exists():
        return str(out), 0, 0
    want = {int(u.rsplit(":", 1)[1]): u for u in uids}
    need_sprint = dict(zip(uids, sprint))
    local = download(path, cache)
    texts = {}
    with open(local, "rb") as fh:
        raw = gzip.GzipFile(fileobj=fh) if local.endswith(".gz") else zstandard.ZstdDecompressor().stream_reader(fh)
        for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
            if i in want:
                texts[want[i]] = json.loads(line).get("text") or ""
                if len(texts) == len(want):
                    break
    missing = set(uids) - set(texts)
    assert not missing, f"{path}: {len(missing)} lines not found"
    tok = tokenizer()
    keys = list(texts)
    rows = dict(doc_uid=[], n_chars=[], n_whole=[], ids_whole=[], n_sprint=[], ids_sprint=[])
    for c in range(0, len(keys), 128):
        part = keys[c:c + 128]
        enc = tok([texts[k] for k in part], add_special_tokens=False).input_ids
        for k, e in zip(part, enc):
            w = np.array(e + [EOS], dtype=np.uint32)
            s = np.array(sprint_ids(tok, texts[k]), dtype=np.uint32) if need_sprint[k] else np.zeros(0, np.uint32)
            rows["doc_uid"].append(k)
            rows["n_chars"].append(len(texts[k]))
            rows["n_whole"].append(len(w))
            rows["ids_whole"].append(w)
            rows["n_sprint"].append(len(s))
            rows["ids_sprint"].append(s)
        del enc
    pd.DataFrame(rows).to_parquet(str(out) + ".tmp")
    os.replace(str(out) + ".tmp", out)
    if delete:
        os.remove(os.path.realpath(local))
    return str(out), len(keys), int(sum(rows["n_whole"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extract_list")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    ex = pd.read_parquet(a.extract_list, columns=["doc_uid", "sprint"])
    path = "data/" + ex.doc_uid.str.rsplit(":", n=1).str[0]
    jobs = [(p, list(d.doc_uid), list(d.sprint), a.cache, a.delete) for p, d in ex.groupby(path)]
    jobs.sort(key=lambda j: -len(j[1]))   # big shards first
    print(f"{len(ex):,} documents in {len(jobs):,} shards", flush=True)
    t0, nd, nt = time.time(), 0, 0
    with ProcessPoolExecutor(a.nproc) as pool:
        for k, (out, n, t) in enumerate(pool.map(shard_job, jobs)):
            nd, nt = nd + n, nt + t
            if k % 25 == 0:
                print(f"{k}/{len(jobs)} shards, {nd:,} docs, {nt / 1e9:.2f}B whole tokens, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"EXTRACT DONE {len(jobs)} shards {nd:,} new docs {nt / 1e9:.2f}B tokens {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
