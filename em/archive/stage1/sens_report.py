"""Report for Phase SENS (sensitivity.py) plus the spectral position of each direction in the activation covariance.

Per direction, each as a ratio to the mean of the uniform random directions (random*) and, separately, to the mean of
the covariance-shaped random directions (covrand*):
  kl_a{alpha}    mean per-token KL(base || steered) when alpha * d is added at every position
  dce_a{alpha}   mean change in next-token loss under the same steering
  slope          first-order loss change per token per unit of steering, E[s.d] / E[n_tok] (signed)
  fisher_doc     E[(s.d)^2] / E[n_tok]: curvature of the pretraining loss along d for a document-level perturbation
  fisher_tok     d' (sum_t g_t g_t^T / n) d: the same for a position-wise perturbation
  var            d' Sigma d (activation variance along d), from prevalence.py's cov.pt
and the paired G - N difference per domain with a document bootstrap CI. Also: where d' Sigma d falls among the
eigenvalues of Sigma, and the share of d's energy in the top 1 / 10 / 100 principal components.
Writes results/sensitivity/<tag>/report.txt.

  python sens_report.py --tag gn_l12
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("med_G", "med_N"), ("fin_G", "fin_N"), ("spo_G", "spo_N")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cov", default="results/prevalence/s1_l12_narrow/cov.pt")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    P = ROOT / "results" / "sensitivity" / args.tag
    meta = json.load(open(P / "meta.json"))
    names, alphas = meta["names"], meta["alphas"]
    D = torch.tensor(meta["D"], dtype=torch.float64)
    shards = sorted(P.glob("docs_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
    sg = [torch.load(P / f"sgrad_{f.stem.split('_')[1]}.pt") for f in shards]
    S = torch.cat([x["s"] for x in sg]).double()
    sdoc = torch.cat([x["doc"] for x in sg])
    df = df.set_index("doc").loc[sdoc.numpy()].reset_index()          # align rows with sgrad
    gr = [torch.load(P / f"gram_{f.stem.split('_')[1]}.pt") for f in shards]
    gram = sum(x["gram"] for x in gr) / sum(x["n"] for x in gr)
    C = torch.load(ROOT / args.cov, map_location="cpu", weights_only=True)["cov"].double()
    ev, U = torch.linalg.eigh(C)
    ev, U = ev.flip(0), U.flip(1)

    w = df.n_tok.to_numpy(float)
    proj = (S @ D.T).numpy()                                            # [docs, dirs]: s . d per document
    metrics = {}
    for lab in [f"a{a:g}" for a in alphas]:
        for q in ("kl", "dce"):
            metrics[f"{q}_{lab}"] = {n: df[f"{q}_{n}_{lab}"].to_numpy() for n in names}
    per_doc = {"slope": proj / w[:, None], "fisher_doc": proj ** 2 / w[:, None]}
    tokquad = {n: float(D[j] @ gram @ D[j]) for j, n in enumerate(names)}
    var = {n: float(D[j] @ C @ D[j]) for j, n in enumerate(names)}

    def wmean(x, ww):
        return float((x * ww).sum() / ww.sum())

    def value(metric, n, idx=None):
        ii = slice(None) if idx is None else idx
        if metric in metrics:
            return wmean(metrics[metric][n][ii], w[ii])
        if metric in per_doc:
            j = names.index(n)
            x = per_doc[metric][:, j] * w                                 # back to per-doc totals
            return float(x[ii].sum() / w[ii].sum())
        return {"fisher_tok": tokquad, "var": var}[metric][n]

    rnd = [n for n in names if n.startswith("random")]
    cov_rnd = [n for n in names if n.startswith("covrand")]
    trained = [n for n in names if n not in rnd and n not in cov_rnd]
    order = [f"kl_a{a:g}" for a in alphas] + [f"dce_a{a:g}" for a in alphas] + ["slope", "fisher_doc", "fisher_tok", "var"]
    lines = [f"Phase SENS ({args.tag}): {len(df)} Dolma documents, {int(w.sum())} target tokens, layer {meta['layer']}.",
             f"{len(rnd)} uniform random directions, {len(cov_rnd)} covariance-shaped random directions. Ratios are to "
             "the mean of each baseline set; 'pct' is the share of uniform random directions with a smaller value.", ""]
    hdr = f"{'metric':12s} {'dir':12s} {'value':>12s} {'x random':>9s} {'pct rnd':>8s} {'x covrand':>10s}"
    for met in order:
        vals = {n: value(met, n) for n in names}
        r_mean = np.mean([vals[n] for n in rnd])
        c_mean = np.mean([vals[n] for n in cov_rnd]) if cov_rnd else np.nan
        lines.append(hdr if met == order[0] else "")
        lines.append(f"[{met}]  random mean {r_mean:.5g} (range {min(vals[n] for n in rnd):.5g} to "
                     f"{max(vals[n] for n in rnd):.5g});  covrand mean {c_mean:.5g}")
        for n in trained:
            v = vals[n]
            pct = np.mean([vals[r] < v for r in rnd])
            lines.append(f"{met:12s} {n:12s} {v:12.5g} {v / r_mean:9.3f} {pct:8.2f} {v / c_mean:10.3f}")

    lines += ["", "Paired G - N per domain (document bootstrap, 95% CI); positive = G larger"]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(df), len(df)) for _ in range(args.n_boot)]
    for met in order[:-2]:
        for g, n in PAIRS:
            diff = value(met, g) - value(met, n)
            bs = [value(met, g, b) - value(met, n, b) for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"{met:12s} {g[:3]}  G - N {diff:+.5g}  [{lo:+.5g}, {hi:+.5g}]")

    lines += ["", "Spectral position in the activation covariance (top eigenvalue %.4g, mean %.4g)" % (ev[0], ev.mean()),
              f"{'dir':12s} {'var':>10s} {'x mean eig':>10s} {'eig rank':>9s} {'top1':>7s} {'top10':>7s} {'top100':>7s}"]
    for j, n in enumerate(trained + cov_rnd[:3] + rnd[:3]):
        d = D[names.index(n)]
        c = (U.T @ d) ** 2
        v = float(d @ C @ d)
        rank = int((ev > v).sum())
        lines.append(f"{n:12s} {v:10.4g} {v / float(ev.mean()):10.3f} {rank:9d} {float(c[:1].sum()):7.3f} "
                     f"{float(c[:10].sum()):7.3f} {float(c[:100].sum()):7.3f}")
    lines.append(f"(uniform random expectation: top1 {1 / len(ev):.4f}, top10 {10 / len(ev):.4f}, top100 {100 / len(ev):.4f})")

    lines += ["", "Across all directions: Spearman correlation of variance share with each metric"]
    allv = np.array([var[n] for n in names])
    for met in order[:-1]:
        m = np.array([value(met, n) for n in names])
        lines.append(f"  var vs {met:12s} rho {pd.Series(allv).corr(pd.Series(m), method='spearman'):+.3f}")
    txt = "\n".join(lines)
    (P / "report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
