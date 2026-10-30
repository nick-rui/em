"""Phase AB follow-up: which pretraining documents does the GENERAL direction matter for more than the NARROW one?
Per document (ablate_loss.py output): d_GN = loss(all-layer G ablated) − loss(all-layer N ablated).
Null for its spread: the same difference between two random-direction ablations (rand_i − rand_j).
Reliability: Spearman of d_GN (all layers) with the sum over single-layer L16/L20/L24 G − N on the 4k docs that have it.
Top / bottom 200 docs by d_GN: stratum over-representation (one-sided hypergeometric, Bonferroni) and snippets. CPU only.
  python em/ablate_docs.py > results/ablate/docs_GminusN.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "ablate"
K = 200


def overrep(d, top):
    N, rows = len(d), []
    for s in sorted(d.stratum.unique()):
        Ks, x = int((d.stratum == s).sum()), int((top.stratum == s).sum())
        rows.append(dict(stratum=s, observed=x, expected=round(len(top) * Ks / N, 1),
                         ratio=round(x / (len(top) * Ks / N), 2), p=hypergeom.sf(x - 1, N, Ks, len(top))))
    r = pd.DataFrame(rows)
    r["p_bonf"] = (r.p * len(r)).clip(upper=1)
    return r.sort_values("p")


def main():
    d = pd.read_parquet(P / "docs.parquet")
    rc = sorted(c for c in d.columns if c.startswith("loss_all_rand"))
    d["d_GN"] = d.loss_all_G - d.loss_all_N
    d["dG"], d["dN"] = d.loss_all_G - d.loss_base, d.loss_all_N - d.loss_base
    d["dR"] = d[rc].mean(1) - d.loss_base
    nulls = np.concatenate([(d[a] - d[b]).to_numpy() for i, a in enumerate(rc) for b in rc[i + 1:]])
    print(f"docs {len(d)}; per-doc G − N (all layers): mean {d.d_GN.mean():+.5f}, sd {d.d_GN.std():.5f}; "
          f"random_i − random_j sd {nulls.std():.5f} (ratio {d.d_GN.std() / nulls.std():.2f})")
    print(f"per-doc |Δloss| sd: G {d.dG.std():.5f}, N {d.dN.std():.5f}, random {d[rc].sub(d.loss_base, axis=0).stack().std():.5f}")
    s = d.dropna(subset=["loss_L16_G"])
    late = sum(s[f"loss_L{L}_G"] - s[f"loss_L{L}_N"] for L in (16, 20, 24))
    early = sum(s[f"loss_L{L}_G"] - s[f"loss_L{L}_N"] for L in (4, 8, 12))
    print(f"reliability on {len(s)} docs: Spearman(all-layer G−N, Σ L16–24 single-layer G−N) {spearmanr(s.d_GN, late)[0]:.3f}; "
          f"with Σ L4–12 {spearmanr(s.d_GN, early)[0]:.3f}; late vs early {spearmanr(late, early)[0]:.3f}")
    print(f"Spearman(dG, dN) per doc {spearmanr(d.dG, d.dN)[0]:.3f}; (dG, dR) {spearmanr(d.dG, d.dR)[0]:.3f}; "
          f"(dN, dR) {spearmanr(d.dN, d.dR)[0]:.3f}; (d_GN, base loss) {spearmanr(d.d_GN, d.loss_base)[0]:.3f}; "
          f"(d_GN, n_tok) {spearmanr(d.d_GN, d.n_tok)[0]:.3f}")
    text = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
    for lab, top in (("G matters MORE than N (top d_GN)", d.nlargest(K, "d_GN")),
                     ("N matters MORE than G (bottom d_GN)", d.nsmallest(K, "d_GN"))):
        r = overrep(d, top)
        sig = r[r.p_bonf < 0.05]
        print(f"\n== {lab}, {K} docs: d_GN {top.d_GN.min():+.4f} … {top.d_GN.max():+.4f}")
        print("significant strata (Bonferroni): " + (", ".join(f"{x.stratum} {x.ratio}× ({x.observed} vs {x.expected})"
                                                       for x in sig.itertuples()) or "none"))
        print(r.head(6).assign(p=r.p.map("{:.1e}".format), p_bonf=r.p_bonf.map("{:.1e}".format)).to_string(index=False))
        for x in top.head(20).itertuples():
            print(f"  {x.d_GN:+.4f} (G {x.dG:+.4f}, N {x.dN:+.4f}, rand {x.dR:+.4f}) {x.stratum:26s} {text[x.doc_uid][:150]!r}")
    d[["doc_uid", "stratum", "weight", "n_tok", "loss_base", "dG", "dN", "dR", "d_GN"]].to_parquet(P / "docs_GminusN.parquet")


if __name__ == "__main__":
    main()
