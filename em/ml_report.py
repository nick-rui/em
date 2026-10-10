"""Phases ML + FB report: how much the model needs each direction to predict pretraining text, ablated across layers
12-24, with interpretable feature directions as the baseline. CPU only.

Δloss = token-weighted mean rise in next-token cross-entropy when the direction is mean-ablated at the layers shown,
over the held-out pool documents; 95% CI by paired bootstrap over documents. "x random" divides by the mean over the
random unit directions ablated the same way.
  python em/ml_report.py > results/ablate/ml/report.txt
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "ablate" / "ml"
B = 1000
GROUPS = [("EM directions", [("sh_em3", "shared(badmed, finance, sports)"), ("sh_perp", "shared EM minus benign"),
                             ("badmed", "bad medical"), ("finance", "risky financial"), ("sports", "extreme sports"),
                             ("f_badmed", "bad vs good medical (difference of means)")]),
          ("benign controls", [("sh_ben", "shared(goodmed, vehicles)"), ("goodmed", "good medical"),
                               ("vehicles", "vehicles")]),
          ("interpretable features (difference of means)",
           [("f_code", "code vs prose"), ("f_academic", "academic vs casual"), ("f_math", "math vs prose"),
            ("f_health", "health vs other"), ("f_politics", "politics vs other")])]


def main():
    d = pd.read_parquet(P / "docs.parquet")
    meta = json.load(open(P / "meta.json"))
    var = json.load(open(P / "variance.json"))
    rng = np.random.default_rng(0)
    w = (d.weight * d.n_tok).to_numpy()

    def delta(col):
        if col not in d:
            return None
        x = (d[col] - d.loss_base).to_numpy()
        ok = ~np.isnan(x)
        if ok.sum() == 0:
            return None
        x, ww = x[ok], w[ok]
        m = float(np.average(x, weights=ww))
        bs = [np.average(x[j], weights=ww[j]) for j in (rng.integers(0, len(x), len(x)) for _ in range(B))]
        return m, *np.percentile(bs, [2.5, 97.5]), int(ok.sum())

    layers = meta["layers"]
    print(f"layers {layers}; {len(d)} documents (single-layer rows use the first {meta['n_docs_single']}); "
          f"base loss {np.average(d.loss_base, weights=w):.4f} nats/token\n")
    rnd = [delta(c)[0] for c in d.columns if c.startswith("loss_all_rand")]
    print(f"ALL LAYERS {layers} ABLATED AT ONCE — random baseline {np.mean(rnd):+.5f} "
          f"({min(rnd):+.5f}–{max(rnd):+.5f}, n={len(rnd)})")
    print(f"  {'direction':20s} {'what it is':44s} {'Δloss':>10s} {'95% CI':>22s} {'x rand':>7s} {'z':>6s}")
    for title, rows in GROUPS:
        print(f"  -- {title}")
        for k, lab in rows:
            r = delta(f"loss_all_{k}")
            if r is None:
                print(f"  {k:20s} {lab:44s} {'(not run)':>10s}")
                continue
            m, lo, hi, _ = r
            print(f"  {k:20s} {lab:44s} {m:+.5f} [{lo:+.5f}, {hi:+.5f}] {m / np.mean(rnd):7.2f} "
                  f"{(m - np.mean(rnd)) / (np.std(rnd) + 1e-12):+6.1f}")

    print("\nPER LAYER (Δloss x random at that layer)")
    keys = [k for _, rows in GROUPS for k, _ in rows]
    hdr = f"  {'direction':20s}" + "".join(f"{'L' + str(L):>9s}" for L in layers) + f"{'var/rand':>10s}"
    print(hdr)
    for k in keys:
        cells, ok = [], False
        for L in layers:
            r = delta(f"loss_L{L}_{k}")
            rr = [delta(f"loss_L{L}_rand{i}") for i in range(3)]
            rr = [x[0] for x in rr if x]
            if r and rr:
                cells.append(f"{r[0] / np.mean(rr):9.2f}")
                ok = True
            else:
                cells.append(f"{'—':>9s}")
        v = np.mean([var[f"L{L}_{k}"] for L in layers if f"L{L}_{k}" in var] or [np.nan])
        if ok:
            print(f"  {k:20s}" + "".join(cells) + f"{v:10.2f}")

    f = ROOT / "results" / "spectrum" / "shared_ml_cos.json"
    if f.exists():
        j = json.load(open(f))
        print("\ncos of the constructed shared directions with the trained vectors they came from")
        for L, row in j.items():
            for n in ("sh_em3", "sh_em3_perp_ben"):
                if n in row:
                    print(f"  L{L:>2s} {n:16s} " + "  ".join(f"{k} {c:+.2f}" for k, c in row[n].items()))


if __name__ == "__main__":
    main()
