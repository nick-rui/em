"""Phase SP report: the GPU pretraining measures on the shared cross-domain EM direction, layer 12. CPU only.
  prevalence  mean-squared projection and tail-token share per document, ratio to the 32 random directions
  salience    mean KL(base || steered) per token at matched norms
  ablation    mean rise in pretraining loss when the direction is mean-ablated at layer 12 (20k held-out pool docs)
Also reprints the layer-4 salience refit on stage 1 (the Phase D4 numbers used the final base model by mistake).
  python em/sp_report.py > results/spectrum/shared_pretraining.txt
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NAMED = ["sh_em3", "sh_em3_perp_ben", "sh_fs", "sh_ben", "res_badmed", "badmed", "badmed_kl3"]
B = 1000


def main():
    rng = np.random.default_rng(0)
    p = ROOT / "results" / "prevalence" / "s1_l12_shared" / "docs.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        rand = [c[4:] for c in d.columns if c.startswith("msq_random")]
        w = d.n_tok.to_numpy()
        print("PREVALENCE at layer 12 (ratio to the mean over 32 random directions)")
        print(f"{'direction':18s} {'msq (variance)':>15s} {'tails':>8s}")
        for col, lab in (("msq", "variance"), ("tail", "tails")):
            pass
        for n in NAMED:
            if f"msq_{n}" not in d:
                continue
            r = {}
            for col, mult in (("msq", w), ("tail", np.ones_like(w))):
                tot = lambda k: (d[f"{col}_{k}"].to_numpy() * mult).sum()
                r[col] = tot(n) / np.mean([tot(k) for k in rand])
            print(f"{n:18s} {r['msq']:15.2f} {r['tail']:8.2f}")
    else:
        print("PREVALENCE: not run")

    f = ROOT / "results" / "significance_s1_l12_shared.parquet"
    if f.exists():
        s = pd.read_parquet(f)
        s["v"] = np.where(s.vec.str.startswith("random"), "random", s.vec)
        print("\nSALIENCE at layer 12: mean KL(base || steered) per token, by steering norm (stage 1)")
        print(s.groupby(["norm", "v"]).kl.mean().unstack().round(4).to_string())
    else:
        print("\nSALIENCE: not run")

    f = ROOT / "results" / "ablate" / "shared" / "docs.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        w = (d.weight * d.n_tok).to_numpy()
        print(f"\nABLATION at layer 12 ({len(d)} held-out pool docs; base loss "
              f"{np.average(d.loss_base, weights=w):.4f}); mean rise in loss, nats/token")
        rows = {}
        for c in [c for c in d.columns if c.startswith("L12_") or c.startswith("loss_L12_")]:
            k = c.split("L12_", 1)[1]
            x = (d[c if c.startswith("loss") else "loss_" + c] - d.loss_base).to_numpy()
            ok = ~np.isnan(x)
            m = float(np.average(x[ok], weights=w[ok]))
            bs = [np.average(x[ok][j], weights=w[ok][j]) for j in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(B))]
            rows[k] = (m, *np.percentile(bs, [2.5, 97.5]))
        rnd = [v[0] for k, v in rows.items() if k.startswith("rand")]
        for k in NAMED:
            if k in rows:
                m, lo, hi = rows[k]
                print(f"  {k:18s} {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  = {m / np.mean(rnd):4.2f}x random  "
                      f"(z {(m - np.mean(rnd)) / (np.std(rnd) + 1e-12):+.1f})")
        if rnd:
            print(f"  {'random':18s} {np.mean(rnd):+.5f}  ({min(rnd):+.5f}-{max(rnd):+.5f}, n={len(rnd)})")
    else:
        print("\nABLATION: not run")

    f = ROOT / "results" / "significance_s1_l4_fixed.parquet"
    if f.exists():
        s = pd.read_parquet(f)
        s["v"] = np.where(s.vec.str.startswith("random"), "random", s.vec)
        print("\nCORRECTION — layer-4 salience refit on STAGE 1 (Phase D4 used the final base model by mistake)")
        print(s.groupby(["norm", "v"]).kl.mean().unstack().round(4).to_string())
        print("  Phase D4 as published (final base): norm 4/8/12 G 0.048/0.168/0.417, N 0.040/0.192/0.582, "
              "random 0.108/0.733/1.725")


if __name__ == "__main__":
    main()
