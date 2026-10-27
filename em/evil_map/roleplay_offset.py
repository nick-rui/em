"""Persona map (em/evil_map/PREREG.md): how much does "being asked to role-play at all" bias the persona points
relative to the fine-tuned checkpoints, which are never asked to role-play? (Nick, 2026-10-27: "is there some vector
of just general role-playing that is biasing all the role-play personas?")

R = mean(the 32 prompted persona points) - mean(the no-persona default points), the direction that separates
"prompted to be someone" from "answering as itself". Reports:
  - |cos(R, e)|, how much of it lies along the aligned -> misaligned axis. Our axis is a difference of two prompted
    groups, so a role-play offset common to both cancels from e; this measures what is left.
  - the constant shift it puts on every unprompted point's position, <R,e>/<e,e>.
  - every final checkpoint's distance to the three group means, before and after projecting R out, against the
    persona groups' own radii -- i.e. how much of the known 1.2-1.9 U gap is this one offset.
Split-half estimator and U units throughout, as group_dist.py.
  python em/evil_map/roleplay_offset.py full [--space generic] [--hidden 24] [--readout resp]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--space", default="generic")
    ap.add_argument("--hidden", default="24")
    ap.add_argument("--readout", default="resp")
    args = ap.parse_args()
    d = ROOT / "data" / "evil_map" / args.name
    meta = json.load(open(d / "meta.json"))
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "personas.jsonl")}
    pa = pd.read_json(d / "answers.jsonl", lines=True)
    pts = meta["points"]
    ckd = sorted(p for p in (d / "ckpt").iterdir() if (p / f"{args.readout}_h{args.hidden}.npy").exists())
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

    h = int(args.hidden)
    X0 = np.load(d / f"{args.readout}_h{h}.npy").astype(np.float32)
    dim = X0.shape[1]
    names = pts + cid
    QM = np.zeros((len(qids), len(names), dim), np.float32)
    n = np.zeros((len(qids), len(names)), np.float32)
    np.add.at(QM, (pa.question_id.map(qi).values, pa.point_id.map({p: k for k, p in enumerate(pts)}).values), X0)
    np.add.at(n, (pa.question_id.map(qi).values, pa.point_id.map({p: k for k, p in enumerate(pts)}).values), 1)
    del X0
    for j, p in enumerate(ckd):
        Y = np.load(p / f"{args.readout}_h{h}.npy").astype(np.float32)
        q = pd.read_json(p / "answers.jsonl", lines=True).question_id.map(qi).values
        np.add.at(QM, (q, np.full(len(Y), len(pts) + j)), Y)
        np.add.at(n, (q, np.full(len(Y), len(pts) + j)), 1)
    QM /= np.maximum(n, 1)[..., None]
    H = (W.reshape(-1, len(qids)) @ QM.reshape(len(qids), -1)).reshape(S_SPLITS, 2, len(names), dim)
    ix = {p: k for k, p in enumerate(names)}

    grp = {t: [p for p in pts if P[p]["trait"] == t] for t in ("aligned", "misaligned", "narrow")}
    persona = [p for p in pts if P[p]["kind"] == "persona"]
    default = [p for p in pts if P[p]["trait"] == "default"]
    G = {t: H[:, :, [ix[p] for p in ps]].mean(2) for t, ps in grp.items()}
    E = G["misaligned"] - G["aligned"]
    U2 = float(np.mean(np.einsum("rd,rd->r", E[:, 0], E[:, 1])))
    # The persona reference must be valence-neutral, or R just records that 22 of the 32 personas are misaligned or
    # narrow and so the persona mean sits up the axis. The aligned and misaligned prompts are exact mirrors, so their
    # midpoint carries the role-play component with the valence cancelled.
    D0 = H[:, :, [ix[p] for p in default]].mean(2)
    Rv = 0.5 * (G["aligned"] + G["misaligned"]) - D0
    Rall = H[:, :, [ix[p] for p in persona]].mean(2) - D0

    # |cos(R, e)| and |R|, from cross-half products so sampling noise does not inflate either norm
    rr = float(np.mean(np.einsum("rd,rd->r", Rv[:, 0], Rv[:, 1])))
    re_ = float(np.mean(0.5 * (np.einsum("rd,rd->r", Rv[:, 0], E[:, 1]) + np.einsum("rd,rd->r", Rv[:, 1], E[:, 0]))))
    ra = float(np.mean(np.einsum("rd,rd->r", Rall[:, 0], Rall[:, 1])))
    rae = float(np.mean(0.5 * (np.einsum("rd,rd->r", Rall[:, 0], E[:, 1]) + np.einsum("rd,rd->r", Rall[:, 1], E[:, 0]))))
    print(f"=== {args.space} space, hidden {h}, readout {args.readout}")
    print(f"  R (valence-neutral: midpoint of the mirrored groups - the {len(default)} no-persona points)")
    print(f"    |R| = {np.sqrt(max(rr, 0) / U2):.2f} U   |cos(R, axis)| = {abs(re_) / np.sqrt(max(rr, 0) * U2):.3f}"
          f"   position shift on an unprompted point: {re_ / U2:+.2f} U")
    print(f"  for comparison, mean of all 32 personas - the no-persona points (confounded by the set's composition)")
    print(f"    |R| = {np.sqrt(max(ra, 0) / U2):.2f} U   |cos(R, axis)| = {abs(rae) / np.sqrt(max(ra, 0) * U2):.3f}"
          f"   position shift: {rae / U2:+.2f} U")

    u = 0.5 * (Rv[:, 0] + Rv[:, 1])                      # one direction, averaged over the halves, for projecting out
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)

    def dist(A, B, drop):
        X, Y = A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]
        if drop:
            X = X - np.einsum("rd,rd->r", X, u)[:, None] * u
            Y = Y - np.einsum("rd,rd->r", Y, u)[:, None] * u
        v = np.mean(np.einsum("rd,rd->r", X, Y))
        return float(np.sqrt(max(v, 0.0) / U2))

    print("\n  persona groups' own radii (how tight each group is), before / after removing R:")
    for t, ps in grp.items():
        v0 = [dist(H[:, :, [ix[p]]].mean(2), G[t], False) for p in ps]
        v1 = [dist(H[:, :, [ix[p]]].mean(2), G[t], True) for p in ps]
        print(f"    {t:11s} {np.mean(v0):.2f} / {np.mean(v1):.2f}")

    runs = {p.name: parse_run(json.load(open(p / "meta.json"))["run"]) for p in ckd}
    run_of = {p.name: json.load(open(p / "meta.json"))["run"] for p in ckd}
    fin = {}
    for c in cid:
        r = runs[c]
        fin.setdefault((r["fam"], r["dataset"]), []).append(c)
    print(f"\n  {'method':26s} {'arm':22s} " + "  ".join(f"{t[:11]:>11s}" for t in grp) + "   (before / after removing R)")
    rows = []
    for (fam, ds), cs in fin.items():
        last = {}
        for c in cs:
            st = int(c.split("@")[1]) if c.split("@")[1].isdigit() else -1
            last.setdefault(run_of[c], (st, c))
            if st > last[run_of[c]][0]:
                last[run_of[c]] = (st, c)
        cs = [c for _, c in last.values()]
        cell = [np.mean([dist(H[:, :, [ix[c]]].mean(2), G[t], dr) for c in cs]) for t in grp for dr in (False, True)]
        rows.append((fam, ds, cell))
    for fam, ds, cell in sorted(rows, key=lambda x: (x[1], x[0])):
        print(f"  {fam_name(fam):26s} {DNAME[ds]:22s} " + "  ".join(f"{cell[2 * k]:5.2f} /{cell[2 * k + 1]:5.2f}" for k in range(len(grp))))


if __name__ == "__main__":
    main()
