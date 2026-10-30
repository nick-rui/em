"""Phase AB report: pretraining-loss cost of mean-ablating G / N vs random directions (ablate_loss.py). CPU only.
Mean Δloss = token-weighted (doc weight x n_tok) mean of per-doc loss minus base; 95% CI by paired bootstrap over docs.
  python em/ablate_report.py > results/ablate/report.txt
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "ablate"
B = 1000


def main():
    d = pd.read_parquet(P / "docs.parquet")
    var = json.load(open(P / "variance.json"))
    meta = json.load(open(P / "meta.json"))
    rng = np.random.default_rng(0)
    w = (d.weight * d.n_tok).to_numpy()

    def delta(col, idx=None):
        idx = np.arange(len(d)) if idx is None else idx
        x = (d[col] - d.loss_base).to_numpy()[idx]
        ww = w[idx]
        ok = ~np.isnan(x)
        x, ww = x[ok], ww[ok]
        m = float((x * ww).sum() / ww.sum())
        bs = [((x[j] * ww[j]).sum() / ww[j].sum()) for j in (rng.integers(0, len(x), len(x)) for _ in range(B))]
        return m, *np.percentile(bs, [2.5, 97.5])

    print(f"docs {len(d)} (single-layer: first {meta['n_docs_single']}); base loss {np.average(d.loss_base, weights=w):.4f}\n")
    print("ALL six layers ablated at once (Δloss, nats/token)")
    rows = {c[9:]: delta(c) for c in d.columns if c.startswith("loss_all_")}
    rnd = [v[0] for k, v in rows.items() if k.startswith("rand")]
    for k in ("G", "N"):
        if k in rows:
            m, lo, hi = rows[k]
            print(f"  {k}: {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  | random mean {np.mean(rnd):+.5f}, range {min(rnd):+.5f}–{max(rnd):+.5f}, "
                  f"z {(m - np.mean(rnd)) / (np.std(rnd) + 1e-12):+.1f}  ({len(rnd)} random)")
    if "loss_all_G" in d and "loss_all_N" in d:
        x = (d.loss_all_G - d.loss_all_N).to_numpy()
        bs = [np.average(x[j], weights=w[j]) for j in (rng.integers(0, len(x), len(x)) for _ in range(B))]
        print(f"  G − N: {np.average(x, weights=w):+.5f} [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]")

    print("\nSINGLE layer (first docs only); variance share along the direction ÷ random expectation in brackets")
    idx = np.where(~d.get(f"loss_L{meta['layers'][0]}_G", pd.Series(np.nan, index=d.index)).isna())[0]
    for L in meta["layers"]:
        cols = {c.split("_", 2)[2]: c for c in d.columns if c.startswith(f"loss_L{L}_")}
        if not cols:
            continue
        r = {k: delta(c, idx) for k, c in cols.items()}
        rv = [v[0] for k, v in r.items() if k.startswith("rand")]
        s = f"  L{L:2d}: " + "  ".join(f"{k} {r[k][0]:+.5f} [{r[k][1]:+.5f}, {r[k][2]:+.5f}] (var {var.get(f'L{L}_{k}', float('nan')):.2f})"
                                    for k in ("G", "N") if k in r)
        print(s + f"  | random {np.mean(rv):+.5f} ({min(rv):+.5f}–{max(rv):+.5f}) (var "
              f"{np.mean([var[f'L{L}_rand{i}'] for i in range(len(rv))]):.2f})" if rv else s)

    if "loss_all_G" in d:
        print("\nBy stratum, all-layer Δloss (G / N / random mean), strata with ≥ 100 docs")
        rc = [c for c in d.columns if c.startswith("loss_all_rand")]
        d["dG"], d["dN"] = d.loss_all_G - d.loss_base, d.loss_all_N - d.loss_base
        d["dR"] = d[rc].mean(1) - d.loss_base
        g = d.groupby("stratum").apply(lambda x: pd.Series({"n": len(x), "G": np.average(x.dG, weights=x.n_tok),
                                                            "N": np.average(x.dN, weights=x.n_tok),
                                                            "rand": np.average(x.dR, weights=x.n_tok)}), include_groups=False)
        g = g[g.n >= 100].sort_values("G", ascending=False)
        print(g.round(5).to_string())
        text = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
        print("\nTop 25 docs by all-layer G Δloss minus random-mean Δloss (attribution preview)")
        d["gx"] = d.dG - d.dR
        for _, r in d.nlargest(25, "gx").iterrows():
            print(f"  {r.gx:+.3f} (G {r.dG:+.3f}, N {r.dN:+.3f}) {r.stratum:28s} {text[r.doc_uid][:160]!r}")


if __name__ == "__main__":
    main()
