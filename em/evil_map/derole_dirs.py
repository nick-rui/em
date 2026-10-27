"""Persona map (em/evil_map/PREREG.md): does removing the role-play offset let us tell "went towards the misaligned
personas" apart from "went towards the narrow personas"? (Nick, 2026-10-27: "accounting for role playing bias doesnt
help? give me a before/after".)

Seen from the untrained model, the misaligned and narrow group means are only ~0.5 U apart but both ~1.3 U away, so
the two target directions are nearly parallel and no arm can be assigned to one rather than the other. Every prompted
persona also carries a common role-play offset R = midpoint(the mirrored groups) - mean(no-persona points), and a
component shared by both targets is exactly what makes them parallel. So subtracting R should spread them.

Reports, before and after subtracting R from every persona point:
  cos(towards misaligned, towards narrow)  -- how separable the two hypotheses are at all;
  per arm, cos with each target and the gap between them -- which one it actually went towards.
R_perp (R orthogonalised against the axis) is the identifiable part; R_full also moves the persona midpoint onto the
untrained model, which assumes the untrained model is exactly neutral between a saint and a villain, so it is shown
only for comparison. Split-half estimator and U units, as group_dist.py.
  python em/evil_map/derole_dirs.py full [--space generic] [--hidden 24]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from place_ckpt import DNAME, S_SPLITS, fam_name, parse_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEEP = ("r32", "r1L23", "full")   # the families whose readouts are not in question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--space", default="generic")
    ap.add_argument("--hidden", default="24")
    args = ap.parse_args()
    d = ROOT / "data" / "evil_map" / args.name
    meta = json.load(open(d / "meta.json"))
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "personas.jsonl")}
    pa = pd.read_json(d / "answers.jsonl", lines=True)
    pts = meta["points"]
    h = int(args.hidden)
    ckd = sorted(p for p in (d / "ckpt").iterdir() if (p / f"resp_h{h}.npy").exists())
    cid = [p.name for p in ckd]

    qids = sorted(pa.question_id.unique())
    qi = {q: k for k, q in enumerate(qids)}
    qdom = pa.drop_duplicates("question_id").set_index("question_id").qdomain.reindex(qids).values
    sel = (qdom == "generic") if args.space == "generic" else (qdom == args.space) if args.space in meta["domains"] else None
    parts = [qdom == "generic", qdom != "generic"] if sel is None else [sel]
    rng = np.random.default_rng(0)
    W = np.zeros((S_SPLITS, 2, len(qids)), np.float32)
    for r in range(S_SPLITS):
        for part in parts:
            ks = rng.permutation(np.flatnonzero(part))
            for hf, half in enumerate((ks[: len(ks) // 2], ks[len(ks) // 2:])):
                W[r, hf, half] += 1 / len(half) / len(parts)

    X0 = np.load(d / f"resp_h{h}.npy").astype(np.float32)
    dim = X0.shape[1]
    names = pts + cid
    QM = np.zeros((len(qids), len(names), dim), np.float32)
    n = np.zeros((len(qids), len(names)), np.float32)
    np.add.at(QM, (pa.question_id.map(qi).values, pa.point_id.map({p: k for k, p in enumerate(pts)}).values), X0)
    np.add.at(n, (pa.question_id.map(qi).values, pa.point_id.map({p: k for k, p in enumerate(pts)}).values), 1)
    del X0
    for j, p in enumerate(ckd):
        Y = np.load(p / f"resp_h{h}.npy").astype(np.float32)
        q = pd.read_json(p / "answers.jsonl", lines=True).question_id.map(qi).values
        np.add.at(QM, (q, np.full(len(Y), len(pts) + j)), Y)
        np.add.at(n, (q, np.full(len(Y), len(pts) + j)), 1)
    QM /= np.maximum(n, 1)[..., None]
    H = (W.reshape(-1, len(qids)) @ QM.reshape(len(qids), -1)).reshape(S_SPLITS, 2, len(names), dim)
    ix = {p: k for k, p in enumerate(names)}

    gs = ("aligned", "misaligned", "narrow")
    G = {t: H[:, :, [ix[p] for p in pts if P[p]["trait"] == t]].mean(2) for t in gs}
    D0 = H[:, :, [ix["default__0"]]].mean(2)
    E = G["misaligned"] - G["aligned"]
    U2 = float(np.mean(np.einsum("rd,rd->r", E[:, 0], E[:, 1])))
    R = 0.5 * (G["aligned"] + G["misaligned"]) - D0
    Eb, Rb = 0.5 * (E[:, 0] + E[:, 1]), 0.5 * (R[:, 0] + R[:, 1])
    Rp = Rb - (np.einsum("rd,rd->r", Rb, Eb) / np.maximum(np.einsum("rd,rd->r", Eb, Eb), 1e-9))[:, None] * Eb

    def cos(A, B):
        num = np.mean(0.5 * (np.einsum("rd,rd->r", A[:, 0], B[:, 1]) + np.einsum("rd,rd->r", A[:, 1], B[:, 0])))
        na = np.mean(np.einsum("rd,rd->r", A[:, 0], A[:, 1]))
        nb = np.mean(np.einsum("rd,rd->r", B[:, 0], B[:, 1]))
        return float(num / np.sqrt(max(na, 1e-12) * max(nb, 1e-12)))

    runs = {p.name: parse_run(json.load(open(p / "meta.json"))["run"]) for p in ckd}
    run_of = {p.name: json.load(open(p / "meta.json"))["run"] for p in ckd}
    fin = {}
    for c in cid:
        r = runs[c]
        if r["fam"] in KEEP:
            fin.setdefault((r["fam"], r["dataset"]), []).append(c)

    print(f"=== {args.space} space, hidden {h}; arms: {', '.join(KEEP)}")
    for tag, sub in (("before (as published)", None), ("after, R off the axis only", Rp), ("after, all of R", Rb)):
        Gs = {t: (G[t] - sub[:, None, :] if sub is not None else G[t]) for t in gs}
        U = {t: Gs[t] - D0 for t in gs}
        sep = cos(U["misaligned"], U["narrow"])
        print(f"\n-- {tag}")
        print(f"   cos(towards misaligned, towards narrow) = {sep:+.3f}"
              f"   -> the two hypotheses are {'indistinguishable' if sep > 0.9 else 'partly separable' if sep > 0.6 else 'separable'}")
        print(f"   {'method':22s} {'arm':22s} {'|move|':>7s} {'->squares':>10s} {'->triangles':>12s} {'gap':>7s}")
        for (fam, ds), cs_ in sorted(fin.items(), key=lambda x: (x[0][1], x[0][0])):
            last = {}
            for c in cs_:
                st = int(c.split("@")[1]) if c.split("@")[1].isdigit() else -1
                last.setdefault(run_of[c], (st, c))
                if st > last[run_of[c]][0]:
                    last[run_of[c]] = (st, c)
            A = H[:, :, [ix[c] for _, c in last.values()]].mean(2) - D0
            mv = np.sqrt(max(float(np.mean(np.einsum("rd,rd->r", A[:, 0], A[:, 1]))), 0) / U2)
            cm, cn = cos(A, U["misaligned"]), cos(A, U["narrow"])
            print(f"   {fam_name(fam):22s} {DNAME[ds]:22s} {mv:7.2f} {cm:+10.2f} {cn:+12.2f} {cn - cm:+7.2f}")


if __name__ == "__main__":
    main()
