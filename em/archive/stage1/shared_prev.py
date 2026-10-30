"""Phase Y-prev: the pretraining analysis, redone on the SHARED cross-domain EM direction instead of the
bad-medical-specific one.

Everything so far (prevalence, salience, ablation, CPT, per-doc attribution) was run on G, which Phase Y showed is
75-80% domain-specific. sh_em3 = normalised mean(G, finance, sports) carries nearly all the EM behaviour, so it is the
direction those measures should have been run on.

Two parts are exact from what we already have, with no GPU:
  - corpus variance along any direction d:  d'Cd / trace(C) * d_model   (cov.pt, 19.8M tokens)
  - per-document MEAN projection, which is LINEAR in d, so it is the matching combination of the stored per-document
    mean_* columns (msq and tail columns are not linear and still need a forward pass).
Per-document z = (mean projection - mean over the 32 random directions) / sd over them, as d4_topics.py; then
Common Crawl topic over-representation in the top/bottom 200 (hypergeometric, Bonferroni).

  python em/shared_prev.py > results/spectrum/shared_prevalence.txt
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "prevalence" / "s1"
EM, BEN = ["G", "finance", "sports"], ["goodmed", "vehicles"]
K = 200


def overrep(d, top, col):
    N, rows = len(d), []
    for s in sorted(d.cc_topic.unique()):
        Ks, x = int((d.cc_topic == s).sum()), int((top.cc_topic == s).sum())
        e = len(top) * Ks / N
        rows.append(dict(topic=s, observed=x, expected=round(e, 1), ratio=round(x / e, 2) if e else None,
                         p=hypergeom.sf(x - 1, N, Ks, len(top))))
    r = pd.DataFrame(rows)
    r["p_bonf"] = (r.p * len(r)).clip(upper=1)
    return r.sort_values("p")


def main():
    meta = json.load(open(P / "meta.json"))
    names = meta["names"]
    D = np.array(meta["D"])
    U = {n: D[names.index(n)] / np.linalg.norm(D[names.index(n)]) for n in names}
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    unit = lambda v: v / np.linalg.norm(v)
    rand = [n for n in names if n.startswith("random")]

    # combinations, with the weights needed to rebuild their per-document mean projection from the stored columns
    combos = {"sh_em3": (EM, 1), "sh_fs": (["finance", "sports"], 1), "sh_ben": (BEN, 1)}
    V, W = {}, {}
    for k, (keys, _) in combos.items():
        s = sum(U[x] * np.sign(cos(U[x], U["G"])) for x in keys)
        V[k], W[k] = unit(s), {x: np.sign(cos(U[x], U["G"])) / np.linalg.norm(s) for x in keys}
    p = V["sh_em3"] - cos(V["sh_em3"], V["sh_ben"]) * V["sh_ben"]
    V["sh_em3_perp_ben"] = unit(p)        # not a combination of stored columns alone -> variance only

    C = torch.load(P / "cov.pt", map_location="cpu", weights_only=True)["cov"].double().numpy()
    trace, dm = float(np.trace(C)), C.shape[0]
    print(f"layer {meta['layer']}, {meta['revision']}, 19.8M tokens\n")
    print("CORPUS VARIANCE along each direction (ratio to a random direction; exact, from the covariance)")
    rr = [float(U[n] @ C @ U[n]) / trace * dm for n in rand]
    for n in ["G", "N", "finance", "sports", "insecure"] + BEN + list(V):
        d = V.get(n, U.get(n))
        print(f"  {n:16s} {float(d @ C @ d) / trace * dm:5.2f}")
    print(f"  {'random (32)':16s} {np.mean(rr):5.2f}  (range {min(rr):.2f}-{max(rr):.2f})")

    d = pd.read_parquet(P / "docs.parquet")
    meta_docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["source", "topic"]).rename_axis("doc").reset_index()
    d = d.merge(meta_docs, on="doc", how="left")
    d["cc_topic"] = d.topic.where(d.source == "common_crawl", "(non-CC)")
    R = d[[f"mean_random{i}" for i in range(len(rand))]]
    mu, sd = R.mean(1), R.std(1)
    for k, w in W.items():
        d[f"mean_{k}"] = sum(d[f"mean_{x}"] * c for x, c in w.items())
        d[f"z_{k}"] = (d[f"mean_{k}"] - mu) / sd
    for n in ("G", "N"):
        d[f"z_{n}"] = (d[f"mean_{n}"] - mu) / sd

    print("\nPER-DOCUMENT mean projection (exact; msq/tails would need a forward pass)")
    print("  spearman between per-doc z:")
    cols = ["z_sh_em3", "z_sh_fs", "z_sh_ben", "z_G", "z_N"]
    print(d[cols].corr("spearman").round(3).to_string())
    print("\n  |z| summary: " + ", ".join(f"{c[2:]} sd {d[c].std():.2f}" for c in cols))

    for k in ("sh_em3", "sh_ben"):
        for lab, top in ((f"TOP {K} by z_{k}", d.nlargest(K, f"z_{k}")), (f"BOTTOM {K} by z_{k}", d.nsmallest(K, f"z_{k}"))):
            r = overrep(d, top, k)
            sig = r[(r.p_bonf < 0.05) & (r.topic != "(non-CC)")]
            print(f"\n== {lab}")
            print("  significant CC topics (Bonferroni): "
                  + (", ".join(f"{x.topic} {x.ratio}x ({x.observed} vs {x.expected})" for x in sig.itertuples()) or "none"))
            print("  " + r.head(5).assign(p=r.p.map("{:.1e}".format), p_bonf=r.p_bonf.map("{:.1e}".format)).to_string(index=False).replace("\n", "\n  "))
            print(f"  source share top ÷ sample: "
                  f"{(top.source.value_counts(normalize=True) / d.source.value_counts(normalize=True)).dropna().round(2).sort_values(ascending=False).head(4).to_dict()}")
    out = ROOT / "results" / "spectrum"
    d[["doc", "source", "topic", "cc_topic", "n_tok"] + cols].to_parquet(out / "shared_docs_z.parquet")
    print(f"\nwrote {out}/shared_docs_z.parquet")


if __name__ == "__main__":
    main()
