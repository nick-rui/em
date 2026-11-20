"""Model diffing (em/mitigation/diffing/REPORT.md): readout of weight_diff.py's pass.

  python em/mitigation/diffing/weight_report.py [--pt data/diffing/weight_diff.pt]
    -> results/mitigation/diffing/weight_report.txt, fig_w_*.png, weight_summary.json

Cosines between updates use only pairs with different seeds, because two fine-tunes with the same seed share their data
order and their stochastic-rounding draws (same-seed pairs across bases are ~2x as similar as different-seed pairs in
layer 0), which would inflate the cross-base number.
"""
import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/mitigation/diffing"
CELLS = ["aw1e5", "aw3e6", "aw1e6", "sgd9e2", "sgd3e2", "sgd3e2x3"]
LAB = {"aw1e5": "AdamW 1e-5", "aw3e6": "AdamW 3e-6", "aw1e6": "AdamW 1e-6", "sgd9e2": "SGD 0.09", "sgd3e2": "SGD 0.03",
       "sgd3e2x3": "SGD 0.03, 3 ep", "aw1e5early": "AdamW 1e-5, layers 0-23", "aw1e5late": "AdamW 1e-5, layers 24-47"}
MAT = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def cosm(G):
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / d[:, None] / d[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "data/diffing/weight_diff.pt"))
    a = ap.parse_args()
    W = torch.load(a.pt, weights_only=False)
    names = W["names"]
    ix = {n: i for i, n in enumerate(names)}
    OUT.mkdir(parents=True, exist_ok=True)
    gram = {k: v.numpy() for k, v in W["gram"].items()}
    layers = sorted({L for L, _ in gram if L >= 0})

    def G(sel_layers=None, kinds=MAT):
        return sum(v for (L, k), v in gram.items() if k in kinds and (sel_layers is None or L in sel_layers))

    lines = []
    P = lines.append
    summ = {}
    # 1. whole-update cosines: cross-base (different seeds) vs within-base (different seeds)
    def pairs(c1, c2, same_ok=False):
        out = []
        for s, t in product(range(3), range(3)):
            if s == t and not same_ok:
                continue
            i, j = ix.get(f"{c1}_s{s}"), ix.get(f"{c2}_s{t}")
            if i is not None and j is not None and (c1 != c2 or s < t):
                out.append((i, j))
        return out

    def mcos(C, pr):
        return float(np.mean([C[i, j] for i, j in pr])) if pr else float("nan")

    blocks = {"all": None, "0-11": range(0, 12), "12-23": range(12, 24), "24-35": range(24, 36), "36-47": range(36, 48)}
    P("Weight-space diffing of the full fine-tunes (theta_ft - theta_base, all q/k/v/o/gate/up/down matrices).")
    P("cos between whole updates; different-seed pairs only (seed means over the 6 / 3 pairs).\n")
    P(f"{'setting':16s} {'block':6s} {'no SDF vs v5':>13s} {'no SDF seeds':>13s} {'v5 seeds':>9s}  "
      f"{'nosdf.SDFv5':>11s} {'v5.SDFv5':>9s}  norm nosdf / v5")
    sdf = ix["sdf_v5"]
    for c in CELLS + ["aw1e5early", "aw1e5late"]:
        for bn, bl in blocks.items():
            Gb = G(bl)
            C = cosm(Gb)
            x = mcos(C, pairs(f"nosdf_{c}", f"v5_{c}"))
            wn, wv = mcos(C, pairs(f"nosdf_{c}", f"nosdf_{c}")), mcos(C, pairs(f"v5_{c}", f"v5_{c}"))
            sn = np.mean([C[ix[f"nosdf_{c}_s{s}"], sdf] for s in range(3)])
            sv = np.mean([C[ix[f"v5_{c}_s{s}"], sdf] for s in range(3)])
            nn = np.mean([np.sqrt(Gb[ix[f"nosdf_{c}_s{s}"]] [ix[f"nosdf_{c}_s{s}"]]) for s in range(3)])
            nv = np.mean([np.sqrt(Gb[ix[f"v5_{c}_s{s}"]] [ix[f"v5_{c}_s{s}"]]) for s in range(3)])
            summ[f"{c}|{bn}"] = dict(cross=x, within_nosdf=wn, within_v5=wv, cos_sdf_nosdf=float(sn),
                                     cos_sdf_v5=float(sv), norm_nosdf=float(nn), norm_v5=float(nv))
            P(f"{LAB[c]:16s} {bn:6s} {x:13.3f} {wn:13.3f} {wv:9.3f}  {sn:11.3f} {sv:9.3f}  {nn:.2f} / {nv:.2f}")
        P("")
    # cross-optimizer cosines on the same base
    P("\ncos between settings on the same base (all layers, different seeds):")
    Ca = cosm(G())
    hdr = "".join(f"{LAB[c][:12]:>13s}" for c in CELLS)
    for arm in ("nosdf", "v5"):
        P(f"  {arm}: {'':14s}{hdr}")
        for c1 in CELLS:
            row = "".join(f"{mcos(Ca, pairs(f'{arm}_{c1}', f'{arm}_{c2}')):13.3f}" for c2 in CELLS)
            P(f"  {'':6s}{LAB[c1]:16s}{row}")
    P("  no SDF (rows) vs v5 (columns):")
    for c1 in CELLS:
        row = "".join(f"{mcos(Ca, pairs(f'nosdf_{c1}', f'v5_{c2}')):13.3f}" for c2 in CELLS)
        P(f"  {'':6s}{LAB[c1]:16s}{row}")
    P(f"\ncos(SDF v5 delta, SDF v5.1 delta) all layers: {Ca[ix['sdf_v5'], ix['sdf_v51']]:.3f}; "
      f"norms v5 {np.sqrt(G()[sdf, sdf]):.2f}, v5.1 {np.sqrt(G()[ix['sdf_v51'], ix['sdf_v51']]):.2f}")
    # per-layer series for the figure
    per_layer = defaultdict(dict)
    for L in layers:
        C = cosm(G([L]))
        Gl = G([L])
        for c in CELLS:
            per_layer[c][L] = dict(cross=mcos(C, pairs(f"nosdf_{c}", f"v5_{c}")),
                                   wn=mcos(C, pairs(f"nosdf_{c}", f"nosdf_{c}")), wv=mcos(C, pairs(f"v5_{c}", f"v5_{c}")),
                                   sdf_n=float(np.mean([C[ix[f"nosdf_{c}_s{s}"], sdf] for s in range(3)])),
                                   sdf_v=float(np.mean([C[ix[f"v5_{c}_s{s}"], sdf] for s in range(3)])),
                                   nn=float(np.mean([np.sqrt(Gl[ix[f"nosdf_{c}_s{s}"], ix[f"nosdf_{c}_s{s}"]]) for s in range(3)])),
                                   nv=float(np.mean([np.sqrt(Gl[ix[f"v5_{c}_s{s}"], ix[f"v5_{c}_s{s}"]]) for s in range(3)])))
        per_layer["sdf"][L] = dict(norm=float(np.sqrt(Gl[sdf, sdf])), norm51=float(np.sqrt(Gl[ix["sdf_v51"], ix["sdf_v51"]])),
                                   cos51=float(C[sdf, ix["sdf_v51"]]))
    # 2. write / read projections on the layer-24 directions (times a random direction's share)
    P("\nShare of each update along the layer-24 directions, as a multiple of a random direction's share "
      "(||u^T dW||^2 / (||dW||^2 / 5120); 1 = chance). Write side for o_proj + down_proj, read side for q/k/v/gate/up.")
    dn = W["dnames"]
    proj = {k: v.numpy() for k, v in W["proj"].items()}
    frob = {k: v.numpy() for k, v in W["frob"].items()}
    def share(sel, side):
        kinds = ("o_proj", "down_proj") if side == "write" else ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
        num = sum(proj[(L, k)] for L in sel for k in kinds if (L, k) in proj)
        den = sum(frob[(L, k)] for L in sel for k in kinds if (L, k) in frob)
        return num / den[:, None] * 5120
    proj_layers = {}
    for side in ("write", "read"):
        for bn, sel in (("20-28", range(20, 29)), ("24", [24]), ("0-23", range(24)), ("24-47", range(24, 48))):
            S = share(sel, side)
            P(f"\n  {side}, layers {bn}:  " + "  ".join(f"{d:>10s}" for d in dn))
            for nm in ["sdf_v5", "sdf_v51"] + [f"{arm}_{c}" for c in CELLS for arm in ("nosdf", "v5")]:
                rows = [ix[nm]] if nm.startswith("sdf") else [ix[f"{nm}_s{s}"] for s in range(3)]
                P(f"  {nm:22s}" + "  ".join(f"{S[rows, j].mean():10.1f}" for j in range(len(dn))))
        proj_layers[side] = {L: share([L], side).tolist() for L in layers}
    # 3. full update at layer-24 down_proj vs the rank-1 updates
    P("\ncos(full update at layer-24 down_proj, rank-1 update b a^T) per rank-1 reference (seed means):")
    r1 = W["r1dot"][(24, "down_proj")]
    fr = W["frob"][(24, "down_proj")].numpy()
    for c in CELLS:
        for arm in ("nosdf", "v5"):
            rows = [ix[f"{arm}_{c}_s{s}"] for s in range(3)]
            vals = {r: float(np.mean([r1[r][i].item() / np.sqrt(fr[i]) for i in rows])) for r in r1}
            agg = {g: np.mean([v for r, v in vals.items() if r.startswith(g + "_")]) for g in ("nosdf", "v5", "v51")}
            P(f"  {LAB[c]:16s} {arm:6s} vs nosdf r1 {agg['nosdf']:+.4f}  v5 r1 {agg['v5']:+.4f}  v5.1 r1 {agg['v51']:+.4f}")
    # 4. SVD concentration and top-subspace overlap
    P("\nUpdate concentration: share of ||dW||^2 in the top 1 / 8 / 64 singular values (seed means), o_proj and down_proj")
    svd = W["svd"]
    conc = defaultdict(dict)
    for (L, k, nm), (S, U, f2) in svd.items():
        s2 = (S.double() ** 2).numpy()
        conc[(L, k)][nm] = (s2[0] / f2, s2[:8].sum() / f2, s2[:64].sum() / f2)
    for k in ("down_proj", "o_proj"):
        P(f"  {k}:   layer  " + "  ".join(f"{LAB[c][:10]:>21s}" for c in CELLS[:5]))
        for L in sorted({L for L, kk in conc if kk == k}):
            cells = []
            for c in CELLS[:5]:
                v = [np.mean([conc[(L, k)][f"{arm}_{c}_s{s}"][1] for s in range(3)]) for arm in ("nosdf", "v5")]
                cells.append(f"{v[0]:.2f} / {v[1]:.2f}")
            P(f"  {'':12s}{L:5d}  " + "  ".join(f"{x:>21s}" for x in cells))
    P("  (top-8 share, no SDF / v5)")
    P("\nTop-16 output-subspace overlap ||U_i^T U_j||_F^2 / 16 (different seeds), down_proj:")
    ov = defaultdict(dict)
    for L in sorted({L for L, kk in conc if kk == "down_proj"}):
        def U(nm):
            return svd[(L, "down_proj", nm)][1].float()
        def mo(c1, c2):
            vals = []
            for s, t in product(range(3), range(3)):
                if s == t:
                    continue
                if c1 == c2 and s > t:
                    continue
                vals.append(float((U(f"{c1}_s{s}").T @ U(f"{c2}_s{t}")).pow(2).sum() / 16))
            return np.mean(vals)
        for c in CELLS[:5]:
            ov[c][L] = dict(cross=mo(f"nosdf_{c}", f"v5_{c}"), wn=mo(f"nosdf_{c}", f"nosdf_{c}"), wv=mo(f"v5_{c}", f"v5_{c}"),
                            sdf_v=float(np.mean([(U(f"v5_{c}_s{s}").T @ U("sdf_v5")).pow(2).sum() / 16 for s in range(3)])),
                            sdf_n=float(np.mean([(U(f"nosdf_{c}_s{s}").T @ U("sdf_v5")).pow(2).sum() / 16 for s in range(3)])))
        P(f"  layer {L:2d}: " + "  ".join(f"{LAB[c]}: x {ov[c][L]['cross']:.2f} (n {ov[c][L]['wn']:.2f}, v {ov[c][L]['wv']:.2f}; "
                                         f"SDF n {ov[c][L]['sdf_n']:.2f} v {ov[c][L]['sdf_v']:.2f})" for c in ("aw1e5", "sgd3e2")))
    open(OUT / "weight_report.txt", "w").write("\n".join(lines) + "\n")
    json.dump(dict(summary=summ, per_layer=per_layer, proj_layers=proj_layers, dnames=dn, names=names,
                   overlap={c: v for c, v in ov.items()}), open(OUT / "weight_summary.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
