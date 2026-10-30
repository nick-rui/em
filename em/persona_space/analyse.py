"""Persona space overnight study, per-model analysis (em/persona_space/PREREG.md, pre-registered 2026-10-24 ~02:23 PT).
For one model's persona vectors (extract.py) and the 12 attribute ratings (ratings.csv), at the primary layer
floor(L/2) - 1 (hidden state index + 1), for each framing / set present:
  A1  PCA (mean-centred): Spearman rho of each top-10 component with the alignment rating; the attribute each top-5
      component tracks most
  A2  per attribute, the mean-difference direction (top minus bottom quartile of the rating) fitted on a random half of
      the personas; the other half's variance share along it, mean of 20 splits; a null from 200 permutations
  C3  out-of-fold alignment scores (2 folds, z-scored within fold), for the between-model comparison
  a layer sweep (8 layers) of A1's best |rho| and A2's alignment share and rank
Also saves steering directions at the primary layer from read S1+S3: the alignment direction (toward low ratings, i.e.
toward evil) and the top-5 PCs, each with the personas' SD along it.
  python em/persona_space/analyse.py --name qwen25_14b_it   # -> results/persona_space/models/<name>.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "data" / "persona_space" / "vectors"
OUT = ROOT / "results" / "persona_space" / "models"
ATTRS = ["alignment", "warmth", "competence", "dominance", "activity", "fantastical", "human", "age", "gender",
         "formality", "humor", "assistant"]
SPLITS, PERMS = 20, 200


def ratings():
    """The mean of the two wordings; for age, wording A only (wording B's reversed scale was not followed: the two
    wordings correlate at -0.40; amendment before any vector analysis)."""
    R = {}
    for r in csv.DictReader(open(ROOT / "em" / "persona_space" / "ratings.csv")):
        v = r["a"] if r["attribute"] == "age" else r["rating"]
        if v not in ("", "None"):
            R[(r["id"], r["attribute"])] = float(v)
    return R


def matrix(ids, R):
    return np.array([[R.get((i, a), np.nan) for a in ATTRS] for i in ids])


def direction(X, y):
    lo, hi = np.nanpercentile(y, [25, 75])
    u = X[y >= hi].mean(0) - X[y <= lo].mean(0)
    return u / np.linalg.norm(u)


def share(X, u):
    Xc = X - X.mean(0)
    return (Xc @ u).var() / Xc.var(0).sum()


def a2(X, Y, rng):
    out = {}
    n = len(X)
    for j, a in enumerate(ATTRS):
        y = Y[:, j]
        ok = ~np.isnan(y)
        Xa, ya = X[ok], y[ok]
        m = len(ya)
        sh = []
        for _ in range(SPLITS):
            p = rng.permutation(m)
            tr, te = p[: m // 2], p[m // 2:]
            sh.append(share(Xa[te], direction(Xa[tr], ya[tr])))
        null = []
        for _ in range(PERMS):
            p = rng.permutation(m)
            tr, te = p[: m // 2], p[m // 2:]
            null.append(share(Xa[te], direction(Xa[tr], rng.permutation(ya[tr]))))
        out[a] = dict(share=float(np.mean(sh)), share_sd=float(np.std(sh)), null99=float(np.percentile(null, 99)),
                      null_mean=float(np.mean(null)), n=int(m))
    order = sorted(ATTRS, key=lambda a: -out[a]["share"])
    for k, a in enumerate(order):
        out[a]["rank"] = k + 1
    return out


def a1(X, Y):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    sc = U * S
    var = S ** 2 / (S ** 2).sum()
    ya = Y[:, 0]
    ok = ~np.isnan(ya)
    rho = [float(spearmanr(sc[ok, k], ya[ok]).statistic) for k in range(10)]
    k = int(np.argmax(np.abs(rho)))
    named = []
    for c in range(5):
        r = []
        for j, a in enumerate(ATTRS):
            okj = ~np.isnan(Y[:, j])
            r.append(spearmanr(sc[okj, c], Y[okj, j]).statistic)
        jj = int(np.nanargmax(np.abs(r)))
        named.append(dict(pc=c + 1, attribute=ATTRS[jj], rho=float(r[jj]), var=float(var[c])))
    return dict(rho=rho, var=var[:10].tolist(), best_pc=k + 1, best_abs_rho=abs(rho[k]),
                top3_pass=bool(max(abs(x) for x in rho[:3]) >= 0.5), named=named), Vt[:5], sc[:, :5]


def oof(X, y):
    """Out-of-fold alignment scores: fold = index parity; direction from the other fold; z-scored within fold."""
    s = np.full(len(X), np.nan)
    idx = np.arange(len(X))
    for f in (0, 1):
        te, tr = idx[idx % 2 == f], idx[idx % 2 != f]
        trk = tr[~np.isnan(y[tr])]
        u = direction(X[trk], y[trk])
        z = (X[te] - X[tr].mean(0)) @ u
        s[te] = (z - z.mean()) / z.std()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    R = ratings()
    rng = np.random.default_rng(0)
    d = V / args.name
    res = dict(name=args.name, cells={})
    scores = {}
    for f in sorted(d.glob("*_S?.npy")):
        cell = f.stem
        ids = json.load(open(f.with_suffix(".json")))["ids"]
        A = np.load(f, mmap_mode="r")
        L = A.shape[1] - 1
        lay = L // 2 - 1
        Y = matrix(ids, R)
        X = np.asarray(A[:, lay + 1], dtype=np.float64)
        r1, _, _ = a1(X, Y)
        r2 = a2(X, Y, rng)
        sweep = []
        for l in sorted(set(int(x) for x in np.linspace(1, L - 1, 8))):
            Xl = np.asarray(A[:, l + 1], dtype=np.float64)
            s1, _, _ = a1(Xl, Y)
            s2 = a2_light(Xl, Y, rng)
            sweep.append(dict(layer=l, best_abs_rho=s1["best_abs_rho"], best_pc=s1["best_pc"], **s2))
        res["cells"][cell] = dict(layers=L, layer=lay, n=len(ids), A1=r1, A2=r2, sweep=sweep)
        scores[cell] = dict(ids=ids, oof=oof(X, Y[:, 0]).tolist())
        print(f"{args.name} {cell}: best PC{r1['best_pc']} |rho| {r1['best_abs_rho']:.2f} (top-3 pass {r1['top3_pass']}); "
              f"alignment share {100 * r2['alignment']['share']:.1f}% rank {r2['alignment']['rank']}/12 "
              f"(null99 {100 * r2['alignment']['null99']:.2f}%)", flush=True)
    # steering directions from read S1 + S3 at the primary layer
    parts = [(np.load(d / f"read_{s}.npy", mmap_mode="r"), json.load(open(d / f"read_{s}.json"))["ids"])
             for s in ("S1", "S3") if (d / f"read_{s}.npy").exists()]
    if parts:
        L = parts[0][0].shape[1] - 1
        lay = L // 2 - 1
        X = np.concatenate([np.asarray(a[:, lay + 1], dtype=np.float64) for a, _ in parts])
        ids = sum((i for _, i in parts), [])
        y = matrix(ids, R)[:, 0]
        ok = ~np.isnan(y)
        u = -direction(X[ok], y[ok])                      # toward evil
        Xc = X - X.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        dirs = np.concatenate([u[None], Vt[:5]])
        sd = (Xc @ dirs.T).std(0)
        np.savez(d / "steer_dirs.npz", dirs=dirs.astype(np.float32), sd=sd.astype(np.float32), layer=lay,
                 names=np.array(["alignment_to_evil"] + [f"pc{k}" for k in range(1, 6)]),
                 mean=X.mean(0).astype(np.float32))
        res["steer"] = dict(layer=lay, sd=sd.tolist(), cos_align_pcs=(Vt[:5] @ u).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{args.name}.json", "w"), indent=1)
    json.dump(scores, open(OUT / f"{args.name}_scores.json", "w"))
    print(f"{args.name} ANALYSE DONE", flush=True)


def a2_light(X, Y, rng, splits=5):
    """The layer sweep's lighter A2: alignment's share and rank (5 splits, no null)."""
    sh = {}
    for j, a in enumerate(ATTRS):
        y = Y[:, j]
        ok = ~np.isnan(y)
        Xa, ya = X[ok], y[ok]
        m = len(ya)
        v = []
        for _ in range(splits):
            p = rng.permutation(m)
            v.append(share(Xa[p[m // 2:]], direction(Xa[p[: m // 2]], ya[p[: m // 2]])))
        sh[a] = float(np.mean(v))
    rank = 1 + sum(sh[a] > sh["alignment"] for a in ATTRS)
    return dict(align_share=sh["alignment"], align_rank=int(rank))


if __name__ == "__main__":
    main()
