"""What the math direction picks out (em/TODO.md, math rescore, 2026-10-27): unique pool documents at both ends of
z_math_mean, and where it disagrees with z_sh_em3. Run on the box holding data/pool20m (CPU), from the repo root:
  ssh lambda-4xh100-2 'cd ~/em-olmo && .venv/bin/python -' < em/scripts/math_rescore_docs.py
Writes results/attribution/math_rescore/{composition.txt, docs.txt}:
  composition  source and common-crawl topic shares in the top / bottom 1% and 0.1% of unique documents, as lift over
               the pool share (1.0 = as common as in the pool), with z_sh_em3's top 1% alongside
  docs         20 documents drawn at random (seed 0) from each of: the top 0.1%, the bottom 0.1%, high math but low
               sh_em3 (z_math_mean >= 2, z_sh_em3 <= 0), low math but high sh_em3 (z_math_mean <= 0, z_sh_em3 >= 2.05);
               first 500 characters each
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
POOL = ROOT / "data" / "pool20m"
OUT = ROOT / "results" / "attribution" / "math_rescore"
Z, ZS = "z_math_mean", "z_sh_em3"

d = pd.read_parquet(POOL / "scores_math.parquet", columns=["doc_uid", "shard", "n_tokens", Z, ZS])
d = d.drop_duplicates(["shard", ZS, "n_tokens"]).reset_index(drop=True)
parts = d.shard.str.split("__").str[0].str.split("-")
d["source"] = parts.str[0]
d["topic"] = np.where(d.source == "common_crawl", parts.str[1], "(" + d.source + ")")
n = len(d)
q = {k: d[Z].quantile(v) for k, v in [("top1", .99), ("top01", .999), ("bot1", .01), ("bot01", .001)]}
groups = {"math top 0.1%": d[Z] >= q["top01"], "math top 1%": d[Z] >= q["top1"],
          "math bottom 1%": d[Z] <= q["bot1"], "math bottom 0.1%": d[Z] <= q["bot01"],
          "sh_em3 top 1%": d[ZS] >= d[ZS].quantile(.99)}
L = [f"{n:,} unique documents; z_math_mean quantiles: 0.1% {q['bot01']:.2f}, 1% {q['bot1']:.2f}, 99% {q['top1']:.2f}, "
     f"99.9% {q['top01']:.2f}", "Lift = share in the group / share in the pool.", ""]
for col, minn in (("source", 1000), ("topic", 20000)):
    base = d[col].value_counts(normalize=True)
    tab = pd.DataFrame({g: d[m][col].value_counts(normalize=True).reindex(base.index).fillna(0) / base
                        for g, m in groups.items()})
    tab.insert(0, "pool share", base)
    tab = tab[d[col].value_counts().reindex(tab.index) >= minn].sort_values("math top 1%", ascending=False)
    L += [f"== {col}", tab.to_string(float_format=lambda x: f"{x:.2f}"), ""]
L.append("median tokens: " + ", ".join(f"{g} {int(d[m].n_tokens.median())}" for g, m in groups.items())
         + f", pool {int(d.n_tokens.median())}")
(OUT / "composition.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))

rng = np.random.default_rng(0)
picks = {"TOP 0.1% ON MATH": d[d[Z] >= q["top01"]], "BOTTOM 0.1% ON MATH": d[d[Z] <= q["bot01"]],
         "HIGH MATH, LOW SH_EM3 (z_math >= 2, z_sh_em3 <= 0)": d[(d[Z] >= 2) & (d[ZS] <= 0)],
         "LOW MATH, HIGH SH_EM3 (z_math <= 0, z_sh_em3 >= 2.05)": d[(d[Z] <= 0) & (d[ZS] >= 2.05)]}
S = []
for lab, g in picks.items():
    s = g.iloc[rng.choice(len(g), size=min(20, len(g)), replace=False)]
    S.append(f"======== {lab}: {len(g):,} unique documents; 20 at random ========")
    for key, gg in s.groupby("shard"):
        t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
        for r in gg.itertuples():
            S.append(f"--- [{r.topic}] z_math {getattr(r, Z):+.2f} z_sh_em3 {getattr(r, ZS):+.2f} {r.n_tokens} tok\n"
                     f"{' '.join(str(t.get(r.doc_uid, ''))[:500].split())}\n")
(OUT / "docs.txt").write_text("\n".join(S))
print(f"wrote {OUT / 'composition.txt'} and {OUT / 'docs.txt'}")
