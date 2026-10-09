"""Phase D4 report (pre-registered in TODO.md): layer-4 G vs N representation in the stage-1 pretraining sample.
  python d4_report.py   -> prints the tables; writes results/prevalence/s1_l4/d4_report.json
"""
import json

import numpy as np
import pandas as pd

from common import ROOT

P = ROOT / "results" / "prevalence" / "s1_l4"
rng = np.random.default_rng(0)
B = 2000


def main():
    out = {}
    d = pd.read_parquet(P / "docs.parquet")
    names = [c[4:] for c in d.columns if c.startswith("msq_")]
    rand = [n for n in names if n.startswith("random")]
    w = d.n_tok.to_numpy()

    def ratios(idx, col):
        tot = {n: (d[f"{col}_{n}"].to_numpy()[idx] * (w[idx] if col == "msq" else 1)).sum() for n in names}
        r = np.mean([tot[n] for n in rand])
        return {n: tot[n] / r for n in names}

    for col, lab in (("msq", "P1 variance share / random"), ("tail", "P2 tail tokens / random")):
        full = ratios(np.arange(len(d)), col)
        boot = [ratios(rng.integers(0, len(d), len(d)), col) for _ in range(B)]
        dif = np.array([b["G"] - b["N"] for b in boot])
        rnd = np.array([[b[n] for n in rand] for b in boot])
        out[lab] = {n: full[n] for n in ("G", "N", "G_perp_N", "N_perp_G")}
        out[lab]["G-N"] = [full["G"] - full["N"], *np.percentile(dif, [2.5, 97.5])]
        out[lab]["random_ratio_range"] = [float(min(full[n] for n in rand)), float(max(full[n] for n in rand))]
        print(f"{lab}: " + "  ".join(f"{n} {full[n]:.3f}" for n in ("G", "N", "G_perp_N", "N_perp_G")) +
              f" | G−N {full['G'] - full['N']:+.3f} [{np.percentile(dif, 2.5):+.3f}, {np.percentile(dif, 97.5):+.3f}]"
              f" | random directions {out[lab]['random_ratio_range'][0]:.2f}–{out[lab]['random_ratio_range'][1]:.2f}")

    s = pd.read_parquet(ROOT / "results" / "significance_s1_l4.parquet")
    s["v"] = np.where(s.vec.str.startswith("random"), "random", s.vec)
    t = s.groupby(["norm", "v"]).kl.mean().unstack()
    print("\nSalience: mean KL(base ‖ steered) per token, by norm\n" + t.round(4).to_string())
    out["significance"] = t.to_dict()
    for nm in sorted(s.norm.unique()):
        g = s[(s.vec == "G") & (s.norm == nm)].set_index("doc").kl
        n = s[(s.vec == "N") & (s.norm == nm)].set_index("doc").kl.reindex(g.index)
        dd = (g - n).to_numpy()
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(B)]
        out[f"significance_G-N_norm{nm:g}"] = [float(dd.mean()), *np.percentile(bs, [2.5, 97.5])]
        print(f"  norm {nm:g}: G−N {dd.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")

    a = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "results" / "attribution" / "s1_l4_grad").glob("part-*.parquet"))])
    sup = {c[8:]: a[c].sum() for c in a.columns if c.startswith("support_")}
    r = np.array([v for k, v in sup.items() if k.startswith("random")])
    z = {k: (sup[k] - r.mean()) / r.std() for k in ("G", "N")}
    out["gradient_alignment_corpus_z"] = z
    print(f"\nUsefulness (gradient alignment, corpus sum z vs {len(r)} random): G {z['G']:+.2f}  N {z['N']:+.2f}")

    if (P / "intervene.parquet").exists():
        iv = pd.read_parquet(P / "intervene.parquet")
        res = {}
        for c in iv.columns:
            if "_" in c and c.split("_", 1)[0] in ("ablate", "plus", "minus"):
                kind, name = c.split("_", 1)
                res.setdefault(kind, {})[name] = float((iv[c] - iv["base"]).mean())
        out["intervene_dloss"] = res
        print("\nUsefulness (intervene, mean Δloss on 3,000 docs; push size 6):")
        for kind, r in res.items():
            rv = [v for k, v in r.items() if k.startswith("random")]
            print(f"  {kind:6s} " + "  ".join(f"{n} {r[n]:+.4f}" for n in ("G", "N", "G_perp_N", "N_perp_G") if n in r) +
                  (f" | random mean {np.mean(rv):+.4f} ({min(rv):+.4f}–{max(rv):+.4f})" if rv else ""))
    json.dump(out, open(P / "d4_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
