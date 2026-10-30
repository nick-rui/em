"""Persona map (em/evil_map/PREREG.md): is the aligned -> misaligned axis really the principal axis of the persona
space, or is the 0.98 cosine with PC1 an artefact of the persona set? 20 of the 32 personas are a mirrored
aligned/misaligned pair set, which makes the dominant direction of variance *be* the axis almost by construction (the
Assistant Axis paper, arXiv:2601.10387, reports a far lower cosine over 275 diverse roles). Three checks, per layer:
  all       PC1 of all 32 persona points vs the axis from all of them -- what the report currently quotes
  held out  PC1 of one half of the aligned/misaligned pairs vs the axis built from the other half
  others    PC1 of the narrow and default points alone vs the axis from the aligned/misaligned ones
If the axis survives the last two, it is not merely the mirroring.
  python em/evil_map/check_axis.py full [--hidden 12,24,36]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def cos1(X, e):
    """|cosine| between the first PC of the points X (centred) and the direction e."""
    Y = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return abs(float(Vt[0] @ e / (np.linalg.norm(Vt[0]) * np.linalg.norm(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--hidden", default="12,24,36")
    args = ap.parse_args()
    d = ROOT / "data" / "evil_map" / args.name
    meta = json.load(open(d / "meta.json"))
    M = np.load(d / "point_mean.npy").astype(np.float32)          # [points, hidden, d]
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "personas.jsonl")}
    ids = meta["points"]
    idx = {p: i for i, p in enumerate(ids)}
    ali = [p for p in ids if P[p]["trait"] == "aligned"]
    mis = [p for p in ids if P[p]["trait"] == "misaligned"]
    oth = [p for p in ids if P[p]["trait"] in ("narrow", "default")]
    per = ali + mis + [p for p in ids if P[p]["trait"] == "narrow"]
    print(f"{len(ali)} aligned, {len(mis)} misaligned, {len(oth)} narrow and default")
    print(f"{'hidden':>7s} {'all':>8s} {'held out':>9s} {'others':>8s}   (|cos| of PC1 with the axis)")
    for h in (int(x) for x in args.hidden.split(",")):
        A, B = M[[idx[p] for p in ali], h], M[[idx[p] for p in mis], h]
        e = B.mean(0) - A.mean(0)
        c_all = cos1(M[[idx[p] for p in per], h], e)
        half = len(ali) // 2
        e_half = B[half:].mean(0) - A[half:].mean(0)              # axis from one half of the pairs
        c_held = cos1(np.vstack([A[:half], B[:half]]), e_half)    # PC1 of the other half
        c_oth = cos1(M[[idx[p] for p in oth], h], e)
        print(f"{h:7d} {c_all:8.2f} {c_held:9.2f} {c_oth:8.2f}")


if __name__ == "__main__":
    main()
