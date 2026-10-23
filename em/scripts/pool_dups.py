"""(Diagnostics of the top-1.5B fall, 2026-10-23; run from the repo root on the box holding both sets.) Repeated copies in the pools: documents in the same shard with identical z (both directions) and token count are
copies (checked by text above). Unique counts per pool, per CPT set, and by quality bucket."""
import numpy as np, pandas as pd
for name, path in (("20M pool", "data/pool20m/scores.parquet"), ("15M pool", "data/pool14m/scores.parquet")):
    s = pd.read_parquet(path, columns=["doc_uid", "shard", "z_sh_em3", "z_sh_em3_perp_ben", "n_tokens"])
    key = s.shard + "|" + s.z_sh_em3.round(5).astype(str) + "|" + s.z_sh_em3_perp_ben.round(5).astype(str) + "|" + s.n_tokens.astype(str)
    s["key"] = key
    u = s.key.nunique()
    s["copies"] = s.groupby("key").key.transform("size")
    print(f"{name}: {len(s):,} scored docs, {u:,} unique ({u / len(s):.1%}); docs with >= 2 copies {np.mean(s.copies >= 2):.1%}; "
          f"copies per unique doc max {s.copies.max()}")
    top = s.nlargest(int(0.036 * len(s)), "z_sh_em3")
    print(f"   top 3.6% by z_sh_em3: {top.key.nunique() / len(top):.1%} unique")
    if name == "20M pool":
        for setname, p in (("1B CPT set", "data/pool20m/cpt_top1b/docs.parquet"), ("mix set", "data/pool20m/cpt_top900_rand100/docs.parquet")):
            d = pd.read_parquet(p, columns=["doc_uid", "cpt_tokens"]).merge(s[["doc_uid", "key"]], on="doc_uid")
            first = ~d.key.duplicated()
            print(f"   {setname}: {len(d):,} docs, {d.key.nunique():,} unique ({d.key.nunique() / len(d):.1%}); unique tokens "
                  f"{d.cpt_tokens[first].sum() / 1e9:.2f}B of {d.cpt_tokens.sum() / 1e9:.2f}B")
    else:
        d = pd.read_parquet("data/pool14m/cpt_ext500/docs.parquet", columns=["doc_uid", "cpt_tokens"]).merge(s[["doc_uid", "key"]], on="doc_uid")
        first = ~d.key.duplicated()
        print(f"   extension: {len(d):,} docs, {d.key.nunique():,} unique ({d.key.nunique() / len(d):.1%}); unique tokens "
              f"{d.cpt_tokens[first].sum() / 1e9:.2f}B of {d.cpt_tokens.sum() / 1e9:.2f}B")
