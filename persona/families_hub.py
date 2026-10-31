"""Is "evil" a hub? For each sub-persona family (persona/families.py), how much do its members share one direction in
Qwen2.5-14B-Instruct, and how close are they to the parent persona vector?

  per family: PC1 share of the members' unit vectors, mean pairwise cosine, mean cosine to the parent, per layer and
  averaged over layers 16-36. With results/persona/families_distinguish.json present, also the residual test: across
  all pairs, internal cosine against pairwise distinguishability (how easily a judge tells the two apart), and each
  family's mean residual (above zero = more unified inside the model than its distinguishability predicts).

  python families_hub.py          # -> results/persona/families_hub.json
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VEC = Path(os.environ.get("ABL_VECTORS", HERE.parent / "results" / "qwen_instruct" / "vectors"))
FAM = json.load(open(HERE / "pool" / "families.json"))["families"]
DIST = HERE.parent / "results" / "persona" / "families_distinguish.json"
LAYERS = range(16, 37)


def load(name):
    f = VEC / f"{name}.pt"
    if not f.exists():
        return None
    d = torch.load(f)
    return d["vec"].float() if "vec" in d else None


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def main():
    out = {"families": {}}
    vecs = {}
    for parent, names in FAM.items():
        P = load(parent)
        members = {n: load(n) for n in names}
        members = {n: v for n, v in members.items() if v is not None}
        vecs.update(members)
        if P is not None:
            vecs[parent] = P
        per = []
        for L in LAYERS:
            M = torch.stack([unit(v[L]) for v in members.values()])
            sv = torch.linalg.svdvals(M)
            C = M @ M.T
            k = len(M)
            per.append(dict(pc1=float(sv[0] ** 2 / (sv ** 2).sum()), pair=float((C.sum() - k) / (k * (k - 1))),
                            parent=float((M @ unit(P[L])).mean()) if P is not None else None))
        avg = {key: float(np.mean([x[key] for x in per])) for key in ("pc1", "pair")}
        avg["parent"] = float(np.mean([x["parent"] for x in per])) if P is not None else None
        out["families"][parent] = dict(n=len(members), **avg, layer24=per[24 - LAYERS.start])
        print(f"{parent:12s} n={len(members):2d}  PC1 share {avg['pc1']:.2f}   mean pairwise cos {avg['pair']:.2f}   "
              f"mean cos to parent {avg['parent'] if avg['parent'] is None else round(avg['parent'], 2)}")
    if DIST.exists():
        pairs = json.load(open(DIST))["pairs"]
        fam_of = {n: p for p, names in FAM.items() for n in names}
        xs, ys, rows = [], [], []
        for r in pairs:
            a, b = r["a"], r["b"]
            if a not in vecs or b not in vecs:
                continue
            c = float(np.mean([float(unit(vecs[a][L]) @ unit(vecs[b][L])) for L in LAYERS]))
            xs.append(r["acc"])
            ys.append(c)
            rows.append((a, b, fam_of.get(a), fam_of.get(b), r["acc"], c))
        slope, icpt = np.polyfit(xs, ys, 1)
        res = {}
        for a, b, fa, fb, acc, c in rows:
            if fa == fb:
                res.setdefault(fa, []).append(c - (slope * acc + icpt))
        out["residual"] = dict(slope=float(slope), intercept=float(icpt), n_pairs=len(rows),
                               by_family={f: dict(mean=float(np.mean(v)), n=len(v),
                                                  se=float(np.std(v, ddof=1) / np.sqrt(len(v)))) for f, v in res.items()})
        print(f"internal cos ~ {slope:+.3f} x distinguishability + {icpt:.3f} over {len(rows)} pairs")
        for f, v in sorted(out["residual"]["by_family"].items(), key=lambda kv: -kv[1]["mean"]):
            within = [acc for a, b, fa, fb, acc, c in rows if fa == fb == f]
            print(f"  {f:12s} mean residual {v['mean']:+.3f} (se {v['se']:.3f}, {v['n']} pairs)   "
                  f"within-family distinguishability {np.mean(within):.2f}")
    json.dump(out, open(HERE.parent / "results" / "persona" / "families_hub.json", "w"), indent=1)


if __name__ == "__main__":
    main()
