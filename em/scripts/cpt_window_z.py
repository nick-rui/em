"""(Diagnostics of the top-1.5B fall, 2026-10-23; run from em/ on the box holding data/pool20m.) Does a document's sh_em3 score hold beyond the first 512 tokens? For a sample of the 1B CPT set and of random pool
documents: the same statistic as proj_score.py (mean layer-12 residual, centred, projected, z against the 32 random
directions) on window A = positions 16-511 (what was scored) and window B = positions 512-1023 (trained on, never
scored), for documents with at least 1,024 tokens. Run from em/ on a GPU box.
  python window_z.py <n per group>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from proj_score import MODEL, S1, load_dirs, load_trunk  # noqa: E402

ROOT = Path("..").resolve()
POOL = ROOT / "data" / "pool20m"


class A:
    prev_tag, directions = "s1_l12_shared", "sh_em3,sh_em3_perp_ben"


def texts_for(df):
    out = {}
    for k, g in df.groupby("shard"):
        t = pd.read_parquet(POOL / "text" / f"{k}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid")
        out.update(t.loc[g.doc_uid].text.to_dict())
    return [out[u] for u in df.doc_uid]


def main():
    n = int(sys.argv[1])
    want, layer, D, mu = load_dirs(A)
    nd = len(want)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=S1)
    model = load_trunk(layer, S1, compile_=False)
    top = pd.read_parquet(POOL / "cpt_top1b" / "docs.parquet", columns=["doc_uid", "shard", "z_sh_em3", "cpt_tokens"])
    tot = top.cpt_tokens.sum()
    print(f"CPT tokens beyond position 512 in the 1B set: {(np.maximum(top.cpt_tokens - 1 - 512, 0)).sum() / tot:.1%} "
          f"(docs reaching 1,024 tokens: {np.mean(top.cpt_tokens >= 1025):.1%})", flush=True)
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", "shard", "z_sh_em3", "n_tokens"])
    groups = {"top 1B set": top.sample(min(4 * n, len(top)), random_state=0),
              "random pool": s.sample(4 * n, random_state=0)}
    rows = []
    for g, df in groups.items():
        texts = texts_for(df)
        enc = tok([t[:9000] for t in texts], add_special_tokens=False).input_ids
        keep = [i for i, e in enumerate(enc) if len(e) >= 1024][:n]
        with torch.inference_mode():
            for b in range(0, len(keep), 32):
                idx = keep[b:b + 32]
                ids = torch.tensor([enc[i][:1024] for i in idx], device="cuda")
                h = model(input_ids=ids, use_cache=False).last_hidden_state.float()
                for wname, lo, hi in (("A", 16, 512), ("B", 512, 1024)):
                    p = (h[:, lo:hi].mean(1) - mu) @ D.T
                    z = ((p[:, :nd] - p[:, nd:].mean(1, keepdim=True)) / p[:, nd:].std(1, keepdim=True)).cpu().numpy()
                    for j, i in enumerate(idx):
                        rows.append(dict(group=g, doc=df.doc_uid.iloc[i], window=wname, z_sh_em3=z[j, 0],
                                         z_perp=z[j, 1], z_pool=df.z_sh_em3.iloc[i]))
        print(f"{g}: {len(keep)} documents with >= 1,024 tokens", flush=True)
    r = pd.DataFrame(rows).pivot_table(index=["group", "doc"], columns="window", values="z_sh_em3").reset_index()
    for g, x in r.groupby("group"):
        print(f"{g:12s}: z window A mean {x.A.mean():+.2f} (sd {x.A.std():.2f}), window B mean {x.B.mean():+.2f} "
              f"(sd {x.B.std():.2f}); corr(A, B) {np.corrcoef(x.A, x.B)[0, 1]:.2f}; share of B >= 2.05: "
              f"{np.mean(x.B >= 2.05):.1%}")
    r.to_parquet(ROOT / "data" / "pool20m" / "window_z.parquet")


if __name__ == "__main__":
    main()
