"""Persona map (em/evil_map/PREREG.md): how far is each fine-tune from the aligned, misaligned and narrow personas as
groups, rather than along whichever two principal components happen to be on screen (Nick, 2026-10-27: "Maybe looking
at PC1 versus PC2 isn't the right way to look at this ... take the mean of all the triangles, squares, and circles,
and then measure distance in this space from each one to the averaged mean persona").

Distances use the same split-half estimator as place_ckpt.py: for independent halves A and B of the questions,
d^2(x, G) = <x_A - G_A, x_B - G_B>, averaged over 20 splits, which removes the sampling noise that makes every
distance look larger than it is. Averaging the squared quantity and taking the root once at the end is less biased
than rooting each split, so these numbers run slightly below place_ckpt.py's "off" and "nearest" columns.
Units of U, the distance between the aligned and misaligned means, so U = 1 by construction.

It also prints two yardsticks without which the distances cannot be read: how far each group's own members sit from
their group mean, and the distances between the group means themselves.
  python em/evil_map/group_dist.py full [--space generic] [--hidden 12,24,36] [--pcs]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from place_ckpt import DNAME, OWN, S_SPLITS, fam_name, parse_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--space", default="generic")
    ap.add_argument("--hidden", default="24")
    ap.add_argument("--pcs", action="store_true", help="also break the gap to the narrow mean down by principal component")
    ap.add_argument("--dirs", action="store_true", help="also report which group each arm moved TOWARDS, not just how far")
    args = ap.parse_args()
    d = ROOT / "data" / "evil_map" / args.name
    meta = json.load(open(d / "meta.json"))
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "personas.jsonl")}
    pa = pd.read_json(d / "answers.jsonl", lines=True)
    pts = meta["points"]
    ckd = sorted(p for p in (d / "ckpt").iterdir() if (p / "meta.json").exists())
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

    pq = pa.question_id.map(qi).values
    pk = pa.point_id.map({p: k for k, p in enumerate(pts)}).values
    cq = {p.name: pd.read_json(p / "answers.jsonl", lines=True).question_id.map(qi).values for p in ckd}
    runs = {p.name: parse_run(json.load(open(p / "meta.json"))["run"]) for p in ckd}
    run_of = {p.name: json.load(open(p / "meta.json"))["run"] for p in ckd}

    for h in (int(x) for x in args.hidden.split(",")):
        X0 = np.load(d / f"resp_h{h}.npy").astype(np.float32)
        dim = X0.shape[1]
        names = pts + cid
        QM = np.zeros((len(qids), len(names), dim), np.float32)
        n = np.zeros((len(qids), len(names)), np.float32)
        np.add.at(QM, (pq, pk), X0)
        np.add.at(n, (pq, pk), 1)
        del X0
        for j, p in enumerate(ckd):
            Y = np.load(p / f"resp_h{h}.npy").astype(np.float32)
            np.add.at(QM, (cq[p.name], np.full(len(Y), len(pts) + j)), Y)
            np.add.at(n, (cq[p.name], np.full(len(Y), len(pts) + j)), 1)
        QM /= np.maximum(n, 1)[..., None]
        H = (W.reshape(-1, len(qids)) @ QM.reshape(len(qids), -1)).reshape(S_SPLITS, 2, len(names), dim)
        ix = {p: k for k, p in enumerate(names)}

        groups = {"aligned": [p for p in pts if P[p]["trait"] == "aligned"],
                  "misaligned": [p for p in pts if P[p]["trait"] == "misaligned"],
                  "narrow": [p for p in pts if P[p]["trait"] == "narrow"]}
        for dm in meta["domains"]:
            groups[f"narrow {dm}"] = [p for p in pts if P[p]["trait"] == "narrow" and P[p]["domain"] == dm]
        G = {g: H[:, :, [ix[p] for p in ps]].mean(2) for g, ps in groups.items()}   # [splits, 2, dim]

        U2 = float(np.mean(np.einsum("rd,rd->r", G["misaligned"][:, 0] - G["aligned"][:, 0],
                                     G["misaligned"][:, 1] - G["aligned"][:, 1])))

        def dist(A, B):   # A, B: [splits, 2, dim] -> distance in U
            v = np.mean(np.einsum("rd,rd->r", A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]))
            return float(np.sqrt(max(v, 0.0) / U2))

        print(f"\n=== {args.space} space, hidden {h}: distance to each group's mean, in units of U "
              f"(U = aligned mean to misaligned mean)")
        print("  yardstick -- how far the groups are from each other:")
        gs = ["aligned", "misaligned", "narrow"]
        for i, g1 in enumerate(gs):
            for g2 in gs[i + 1:]:
                print(f"    {g1} mean to {g2} mean: {dist(G[g1], G[g2]):.2f}")
        print("  yardstick -- how far a group's own members sit from their group mean (mean [min, max]):")
        for g, ps in groups.items():
            if len(ps) < 2:
                continue
            v = [dist(H[:, :, [ix[p]]].mean(2), G[g]) for p in ps]
            print(f"    {g:18s} n={len(ps):2d}  {np.mean(v):.2f} [{min(v):.2f}, {max(v):.2f}]")
        du = {g: dist(H[:, :, [ix['default__0']]].mean(2), G[g]) for g in gs}
        print(f"  untrained model: aligned {du['aligned']:.2f}, misaligned {du['misaligned']:.2f}, narrow {du['narrow']:.2f}")

        fin = {}
        for c in cid:
            r = runs[c]
            fin.setdefault((r["fam"], r["dataset"]), []).append(c)
        rows = []
        for (fam, ds), cs in fin.items():
            last = {}
            for c in cs:                      # the final checkpoint of each seed
                st = int(c.split("@")[1]) if c.split("@")[1].isdigit() else -1
                last.setdefault(run_of[c], (st, c))
                if st > last[run_of[c]][0]:
                    last[run_of[c]] = (st, c)
            cs = [c for _, c in last.values()]
            own = f"narrow {OWN[ds]}" if f"narrow {OWN[ds]}" in groups else None
            v = {g: np.mean([dist(H[:, :, [ix[c]]].mean(2), G[g]) for c in cs]) for g in gs}
            vo = np.mean([dist(H[:, :, [ix[c]]].mean(2), G[own]) for c in cs]) if own else np.nan
            rows.append((fam, ds, len(cs), v["aligned"], v["misaligned"], v["narrow"], vo))
        print(f"  {'method':26s} {'arm':22s} {'n':>2s} {'aligned':>8s} {'misaligned':>11s} {'narrow':>7s} {'own-domain narrow':>18s}")
        for fam, ds, k, a, m, nw, vo in sorted(rows, key=lambda x: (x[1], x[0])):
            print(f"  {fam_name(fam):26s} {DNAME[ds]:22s} {k:2d} {a:8.2f} {m:11.2f} {nw:7.2f} {vo:18.2f}")

        if args.dirs:
            # Direction, not distance: each arm's displacement from the untrained model, against the displacement
            # from the untrained model to each group mean. cos = +1 means it set off straight at that group.
            D0 = H[:, :, [ix["default__0"]]].mean(2)
            U = {g: G[g] - D0 for g in gs}
            def cs(A, B):
                num = np.mean(0.5 * (np.einsum("rd,rd->r", A[:, 0], B[:, 1]) + np.einsum("rd,rd->r", A[:, 1], B[:, 0])))
                na = np.mean(np.einsum("rd,rd->r", A[:, 0], A[:, 1]))
                nb = np.mean(np.einsum("rd,rd->r", B[:, 0], B[:, 1]))
                return float(num / np.sqrt(max(na, 1e-12) * max(nb, 1e-12))), float(np.sqrt(max(na, 0) / U2))
            print(f"\n  which group did it set off towards? cos of (arm - untrained) with (group mean - untrained)")
            print(f"  {'method':26s} {'arm':22s} {'|move|':>7s} " + " ".join(f"{g[:10]:>10s}" for g in gs))
            for (fam, ds), cs_ in sorted(fin.items(), key=lambda x: (x[0][1], x[0][0])):
                last = {}
                for c in cs_:
                    st = int(c.split("@")[1]) if c.split("@")[1].isdigit() else -1
                    last.setdefault(run_of[c], (st, c))
                    if st > last[run_of[c]][0]:
                        last[run_of[c]] = (st, c)
                A = H[:, :, [ix[c] for _, c in last.values()]].mean(2) - D0
                vals = [cs(A, U[g]) for g in gs]
                print(f"  {fam_name(fam):26s} {DNAME[ds]:22s} {vals[0][1]:7.2f} " + " ".join(f"{v[0]:+10.2f}" for v in vals))


if __name__ == "__main__":
    main()
