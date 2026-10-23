"""(Diagnostics of the top-1.5B fall, 2026-10-23; run from the repo root on the box holding both sets.) Compare the 1B CPT set with the 500M extension: composition, z, and duplication at matched size."""
import numpy as np, pandas as pd
a = pd.read_parquet("data/pool20m/cpt_top1b/profile.parquet")
b = pd.read_parquet("data/pool14m/cpt_ext500/profile.parquet")
a_m = a[a.z_sh_em3 >= b.z_sh_em3.min()]
sets = {"1B set (all)": a, "1B set, z >= ext cutoff": a_m, "extension": b}
print("== z and length")
for k, x in sets.items():
    print(f"  {k:26s} n {len(x):9,}  z mean {x.z_sh_em3.mean():.2f} median {x.z_sh_em3.median():.2f} min {x.z_sh_em3.min():.2f}  "
          f"median chars {x.n_chars.median():,.0f}")
print("== source (%)")
print(pd.DataFrame({k: x.source.value_counts(normalize=True) * 100 for k, x in sets.items()}).round(2).fillna(0).to_string())
print("== Common Crawl topic (% of CC docs), largest differences ext vs 1B set")
t = pd.DataFrame({k: x[x.source == "common_crawl"].topic.value_counts(normalize=True) * 100 for k, x in sets.items()}).fillna(0)
t["ext - 1B"] = t["extension"] - t["1B set (all)"]
print(t.reindex(t["ext - 1B"].abs().sort_values(ascending=False).index).round(2).head(10).to_string())
print("== Common Crawl quality bucket (%)")
print(pd.DataFrame({k: x[x.source == "common_crawl"].bucket.value_counts(normalize=True) * 100 for k, x in sets.items()}).round(1).fillna(0).sort_index().to_string())

def dup(x):
    p = x.h_prefix[x.h_prefix != 0]
    bd = x.h_body[x.h_body != 0]
    mh = x[[f"mh{i}" for i in range(16)]].to_numpy()
    cand = np.zeros(len(x), bool)
    for band in range(4):   # LSH: 4 bands of 4 hashes; a shared band means a likely near duplicate (Jaccard ~0.7+)
        key = pd.Series([hash(r) for r in map(tuple, mh[:, 4 * band:4 * band + 4])])
        cand |= key.duplicated(keep=False).to_numpy()
    return p.duplicated(keep=False).mean(), bd.duplicated(keep=False).mean(), cand.mean()

rng = np.random.default_rng(0)
print("== duplication within a set, at the extension's size (share of docs with a duplicate)")
for k, x in (("1B set (random 593,821)", a.sample(len(b), random_state=0)), ("1B set, z >= ext cutoff", a_m), ("extension", b)):
    p, bd, c = dup(x)
    print(f"  {k:26s} n {len(x):8,}  same first 500 chars {p:.2%}  same chars 1000-1500 {bd:.2%}  MinHash near-dup {c:.2%}")
p, bd, c = dup(a)
print(f"  {'1B set (all 1.2M)':26s} n {len(a):8,}  same first 500 chars {p:.2%}  same chars 1000-1500 {bd:.2%}  MinHash near-dup {c:.2%}")
ov = b.h_prefix.isin(set(a.h_prefix[a.h_prefix != 0])) & (b.h_prefix != 0)
print(f"== extension docs whose first 500 chars also occur in the 1B set: {ov.mean():.2%}")
