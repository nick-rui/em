"""Rescore the 20M pool's stored layer-12 document means on new directions, on CPU (em/TODO.md, the math direction,
pre-registered 2026-10-27). Same statistic as proj_score.py: proj = (document mean - mu) . d for a unit direction d,
z-scored per document against the 32 random directions of results/prevalence/<prev_tag> (the set z_sh_em3 used), so
the new z share z_sh_em3's scale. sh_em3 itself is rescored too, as a check against the stored z_sh_em3.
Run from the repo root on the box holding data/pool20m (means/ and scores.parquet).
  python em/scripts/rescore_means.py --dir math=runs/s1_math_steer_l12/steering_vector.pt [--dir name=path ...] \
      --out data/pool20m/scores_math.parquet --workers 32
A path is a steering_vector.pt (train.py), a .npy vector, "meta:<name>" (a direction of the prev_tag set), or
"mean:<name>,<name>" (the unit mean of earlier --dir vectors after unit-normalising each, sign-aligned to the first).
--limit N scores only the first N shards (a test).
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
W = MU = NAMES = None


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def load_vec(path, got, meta):
    if path.startswith("meta:"):
        return unit(meta["D"][meta["names"].index(path[5:])])
    if path.startswith("mean:"):
        vs = [got[k] for k in path[5:].split(",")]
        return unit(sum(np.sign(v @ vs[0]) * v for v in vs))
    if path.endswith(".npy"):
        return unit(np.load(ROOT / path))
    return unit(torch.load(ROOT / path, map_location="cpu", weights_only=True)["vector"].float().numpy())


def init(w, mu, names):
    global W, MU, NAMES
    W, MU, NAMES = w, mu, names


def shard(f):
    x = np.load(f).astype(np.float32)
    proj = (x - MU) @ W.T                                      # [n, K]
    nd = len(NAMES)
    r = proj[:, nd:]
    z = (proj[:, :nd] - r.mean(1, keepdims=True)) / r.std(1, ddof=1, keepdims=True)
    out = pd.read_parquet(f[:-4] + ".uids.parquet")
    for k, n in enumerate(NAMES):
        out[f"z_{n}"], out[f"proj_{n}"] = z[:, k], proj[:, k]
    out["shard"] = Path(f).stem
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="name=path (see the docstring)")
    ap.add_argument("--pool", default="data/pool20m")
    ap.add_argument("--prev_tag", default="s1_l12_shared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pdir = ROOT / "results" / "prevalence" / args.prev_tag
    meta = json.load(open(pdir / "meta.json"))
    D = np.array(meta["D"], dtype=np.float64)
    got = {}
    for spec in args.dir:
        name, path = spec.split("=", 1)
        got[name] = load_vec(path, got, meta)
    got["sh_em3_check"] = unit(D[meta["names"].index("sh_em3")])
    rand = D[[i for i, n in enumerate(meta["names"]) if n.startswith("random")]]
    names = list(got)
    sims = {a: {b: round(float(got[a] @ got[b]), 4) for b in names} for a in names}
    print("cosines between the scored directions:", json.dumps(sims))
    w = np.vstack([got[n] for n in names] + [rand]).astype(np.float32)
    mu = torch.load(pdir / "cov.pt", weights_only=True)["mu"].float().numpy()
    files = sorted(glob.glob(str(ROOT / args.pool / "means" / "*.npy")))[:args.limit or None]
    print(f"{len(files)} shards, {len(names)} directions + {len(rand)} random, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(w, mu, names)) as ex:
        parts = list(ex.map(shard, files, chunksize=4))
    df = pd.concat(parts, ignore_index=True)
    old = pd.read_parquet(ROOT / args.pool / "scores.parquet", columns=["doc_uid", "n_tokens", "z_sh_em3"])
    df = df.merge(old, on="doc_uid", how="inner" if args.limit else "left")
    c = np.corrcoef(df.z_sh_em3_check, df.z_sh_em3)[0, 1]
    print(f"check: rescored z_sh_em3 vs stored, corr {c:.6f}, max |dz| {(df.z_sh_em3_check - df.z_sh_em3).abs().max():.4f}"
          f", {df.z_sh_em3.isna().sum()} documents without a stored score")
    df.to_parquet(ROOT / args.out, index=False)
    print(f"wrote {args.out}: {len(df):,} documents")


if __name__ == "__main__":
    main()
