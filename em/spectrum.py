"""Phase X1 (TODO.md): where do the EM directions sit in the layer-12 pretraining activation spectrum, and how do the
EM directions of different fine-tuning domains relate to each other? CPU only.

Inputs (workstream-1 prevalence run, layer 12, stage 1, 19.8M tokens): results/prevalence/s1/{cov.pt, meta.json} —
the centred activation covariance C and the unit directions (EM from bad medical G / narrow N, EM from other domains
finance / sports / insecure, benign goodmed / vehicles / aligned, 32 random).

Per direction d:
  var_ratio   = (d'Cd / trace C) * d_model      share of activation variance along d, 1.0 = a random direction
  top{k}      = sum_{i<k} (d.u_i)^2             share of d inside the top-k principal components
  rank50      = smallest k with top{k} >= 0.5   how deep into the spectrum d's mass reaches
  pr          = (sum (d.u_i)^2)^2 / sum (d.u_i)^4   participation ratio: effective number of PCs d spreads over
Plus the pairwise cosine matrix of the named directions, and the top PCs each direction loads on.

  python em/spectrum.py > results/spectrum/report.txt
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "prevalence" / "s1"
KS = (1, 10, 50, 200, 1000)


def main():
    meta = json.load(open(P / "meta.json"))
    blob = torch.load(P / "cov.pt", map_location="cpu", weights_only=True)
    C = blob["cov"].double()
    d_model = C.shape[0]
    names = meta["names"]
    D = torch.tensor(meta["D"]).double()
    D = D / D.norm(dim=1, keepdim=True)
    named = [n for n in names if not n.startswith("random")]
    rand = [n for n in names if n.startswith("random")]
    idx = {n: i for i, n in enumerate(names)}

    lam, U = torch.linalg.eigh(C)                 # ascending
    lam, U = lam.flip(0), U.flip(1)               # descending
    tot = float(lam.sum())
    print(f"layer {meta['layer']}, {meta['revision']}, {blob['n_tok']:,} tokens, d_model {d_model}")
    print(f"spectrum: top-1 PC {100 * float(lam[0]) / tot:.1f}% of variance, top-10 {100 * float(lam[:10].sum()) / tot:.1f}%, "
          f"top-50 {100 * float(lam[:50].sum()) / tot:.1f}%, top-200 {100 * float(lam[:200].sum()) / tot:.1f}%, "
          f"top-1000 {100 * float(lam[:1000].sum()) / tot:.1f}%\n")

    rows = {}
    for n in names:
        d = D[idx[n]]
        w = (U.T @ d) ** 2                        # mass of d per PC (sums to 1)
        c = torch.cumsum(w, 0)
        rows[n] = dict(var_ratio=float(d @ C @ d) / tot * d_model,
                       **{f"top{k}": float(c[k - 1]) for k in KS},
                       rank50=int((c >= 0.5).nonzero()[0, 0]) + 1,
                       pr=float(1.0 / (w ** 2).sum()),
                       peak=[int(i) + 1 for i in w.topk(5).indices])

    def summ(label, keys):
        v = {k: np.array([rows[n][k] for n in keys]) for k in ("var_ratio", *[f"top{k}" for k in KS], "rank50", "pr")}
        return label, {k: (x.mean(), x.std()) for k, x in v.items()}

    hdr = f"{'direction':12s} {'var/rand':>8s} " + " ".join(f"{'top' + str(k):>7s}" for k in KS) + f" {'rank50':>7s} {'eff.PCs':>8s}"
    print(hdr)
    for n in named:
        r = rows[n]
        print(f"{n:12s} {r['var_ratio']:8.2f} " + " ".join(f"{r['top' + str(k)]:7.3f}" for k in KS)
              + f" {r['rank50']:7d} {r['pr']:8.0f}")
    _, rs = summ("random", rand)
    print(f"{'random(32)':12s} {rs['var_ratio'][0]:8.2f} " + " ".join(f"{rs['top' + str(k)][0]:7.3f}" for k in KS)
          + f" {rs['rank50'][0]:7.0f} {rs['pr'][0]:8.0f}   (sd: var {rs['var_ratio'][1]:.2f}, "
          + ", ".join(f"top{k} {rs['top' + str(k)][1]:.3f}" for k in KS) + ")")

    print("\ntop-5 PCs each direction loads on (PC rank, 1 = largest variance)")
    for n in named:
        print(f"  {n:12s} {rows[n]['peak']}")

    print("\npairwise |cos| of the named directions")
    print(f"{'':12s}" + "".join(f"{n[:9]:>10s}" for n in named))
    for a in named:
        print(f"{a:12s}" + "".join(f"{abs(float(D[idx[a]] @ D[idx[b]])):10.2f}" for b in named))
    rr = [abs(float(D[idx[a]] @ D[idx[b]])) for i, a in enumerate(rand) for b in rand[i + 1:]]
    print(f"random-pair |cos|: mean {np.mean(rr):.3f}, max {np.max(rr):.3f}")

    out = ROOT / "results" / "spectrum"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "eigenvalues_top1000": [float(x) for x in lam[:1000]], "trace": tot,
               "cos": {a: {b: float(D[idx[a]] @ D[idx[b]]) for b in named} for a in named}},
              open(out / "spectrum_l12.json", "w"), indent=1)
    torch.save({"U_top": U[:, :64].float(), "eigenvalues": lam[:64].float(), "layer": meta["layer"]}, out / "pcs_l12.pt")
    print(f"\nwrote {out}/spectrum_l12.json and pcs_l12.pt (top-64 PCs, for steering)")


if __name__ == "__main__":
    main()
