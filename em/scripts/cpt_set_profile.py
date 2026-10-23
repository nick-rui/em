"""(Diagnostics of the top-1.5B fall, 2026-10-23.) Per-document profile of a CPT set for comparing the 1B set with the extension: z, source, topic, quality bucket,
length, and hashes of normalized text windows for duplicate counting (first 500 normalized characters; characters
1000-1500; and a 5-shingle MinHash-style signature of the first 2,000 characters for near duplicates).
  python set_profile.py <pool dir> <set dir> <out parquet>
"""
import hashlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

POOL, SET, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
NPERM = 16
_A = np.random.default_rng(0).integers(1, 2**61 - 1, NPERM, dtype=np.uint64)
_B = np.random.default_rng(1).integers(0, 2**61 - 1, NPERM, dtype=np.uint64)


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def h(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little") if s else 0


def minhash(t):
    w = t.split()
    sh = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
    x = np.array([h(s) for s in sh], dtype=np.uint64)
    return [int(((x * a + b) % np.uint64(2**61 - 1)).min()) for a, b in zip(_A, _B)]


def one(args):
    key, uids = args
    t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text", "source", "topic", "bucket", "n_chars"])
    t = t.set_index("doc_uid").loc[uids]
    rows = []
    for u, r in t.iterrows():
        n = norm(r.text[:3000])
        rows.append(dict(doc_uid=u, source=r.source, topic=r.topic, bucket=r.bucket, n_chars=r.n_chars,
                         h_prefix=h(n[:500]), h_body=h(n[1000:1500]), mh=minhash(n[:2000])))
    return rows


def main():
    d = pd.read_parquet(SET / "docs.parquet", columns=["doc_uid", "z_sh_em3", "shard"])
    with ProcessPoolExecutor(48) as ex:
        rows = [r for part in ex.map(one, d.groupby("shard").doc_uid.apply(list).items(), chunksize=4) for r in part]
    out = d.merge(pd.DataFrame(rows), on="doc_uid")
    mh = np.array(out.pop("mh").tolist(), dtype=np.uint64)
    for i in range(NPERM):
        out[f"mh{i}"] = mh[:, i]
    out.to_parquet(OUT)
    print(len(out), "docs ->", OUT)


if __name__ == "__main__":
    main()
