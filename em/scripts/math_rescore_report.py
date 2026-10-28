"""Report on the pool rescored on the math direction (em/TODO.md, pre-registered 2026-10-27 ~20:10 PT), run on the box
holding data/pool20m (CPU). Unique documents only (OLMo 3 stores upsampled documents as adjacent copies: same shard, z
and length count once), except where the CPT selection rule counts copies.
  1. corr(z_<dir>, z_sh_em3) for each math direction;
  2. the top-1B-token set on z_<dir> under the TOP arm's rule (highest z first, copies included, each document cut to
     1,024 tokens, until 1B tokens), its z cutoff, and its overlap with the TOP set (data/pool20m/cpt_top1b/docs.parquet);
  3. mean z by source and by common-crawl topic, next to z_sh_em3;
  4. samples: the 25 highest unique documents and 25 just above the cutoff, first 700 characters each.
  python em/scripts/math_rescore_report.py [--dir math_mean]   # -> results/attribution/math_rescore/{report.txt,samples.txt}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "pool20m"
OUT = ROOT / "results" / "attribution" / "math_rescore"
BUDGET, CUT = 1_000_000_000, 1024


def top_set(df, col):
    """The TOP arm's rule on column col: highest first, copies included, 1,024-token cut, until 1B tokens."""
    d = df.sort_values(col, ascending=False)
    tok = np.minimum(d.n_tokens.to_numpy(), CUT) + 1                    # + EOS, as pool20m_cpt_data.py
    k = int(np.searchsorted(np.cumsum(tok), BUDGET)) + 1
    return d.iloc[:k], float(d[col].iloc[k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="math_mean")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(POOL / "scores_math.parquet")
    zc = f"z_{args.dir}"
    uniq = df.drop_duplicates(["shard", "z_sh_em3", "n_tokens"])
    L = [f"Pool rescored on the math direction: {len(df):,} rows, {len(uniq):,} unique documents; selection on {zc}", ""]
    for c in [c for c in df.columns if c.startswith("z_math")]:
        L.append(f"corr({c}, z_sh_em3) over unique documents: {np.corrcoef(uniq[c], uniq.z_sh_em3)[0, 1]:.3f}")
    sel, cut = top_set(df, zc)
    su = sel.drop_duplicates(["shard", "z_sh_em3", "n_tokens"])
    top = pd.read_parquet(POOL / "cpt_top1b" / "docs.parquet")
    tcol = "doc_uid" if "doc_uid" in top else top.columns[0]
    ov = su.doc_uid.isin(set(top[tcol])).mean()
    L += ["", f"top 1B tokens on {zc}: {len(sel):,} rows ({len(su):,} unique documents), cutoff z >= {cut:.2f}",
          f"  share of its unique documents also in the TOP set: {ov:.1%}",
          f"  mean z_sh_em3 of the selection {su.z_sh_em3.mean():+.2f} (TOP rule cutoff 2.05; pool mean {uniq.z_sh_em3.mean():+.2f})",
          f"  share of its unique documents with z_sh_em3 >= 2.05: {(su.z_sh_em3 >= 2.05).mean():.1%}"]
    folder = uniq.shard.str.split("__").str[0]
    parts = folder.str.split("-")
    uniq = uniq.assign(source=parts.str[0], topic=np.where(parts.str[0] == "common_crawl", parts.str[1], ""),
                       sel=uniq.doc_uid.isin(set(su.doc_uid)))
    L += ["", f"{'source':28s} {'n':>10s} {'mean ' + zc:>16s} {'mean z_sh_em3':>14s} {'share selected':>15s}"]
    for k, g in uniq.groupby("source"):
        if len(g) >= 1000:
            L.append(f"{k:28s} {len(g):10,d} {g[zc].mean():16.2f} {g.z_sh_em3.mean():14.2f} {g.sel.mean():15.2%}")
    cc = uniq[uniq.source == "common_crawl"]
    L += ["", f"{'common-crawl topic':28s} {'n':>10s} {'mean ' + zc:>16s} {'mean z_sh_em3':>14s} {'share selected':>15s}"]
    for k, g in sorted(cc.groupby("topic"), key=lambda kv: -kv[1][zc].mean()):
        if len(g) >= 1000:
            L.append(f"{k:28s} {len(g):10,d} {g[zc].mean():16.2f} {g.z_sh_em3.mean():14.2f} {g.sel.mean():15.2%}")
    (OUT / "report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    # samples: read the text of the chosen documents from their shards' text files
    hi = su.head(25)
    near = su[su[zc] <= cut + 0.05].head(25)
    S = []
    for lab, part in (("HIGHEST", hi), (f"JUST ABOVE THE CUTOFF ({cut:.2f})", near)):
        S.append(f"======== {lab} ========")
        for key, g in part.groupby("shard"):
            t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid")
            for r in g.itertuples():
                txt = t.text.get(r.doc_uid, "")
                S.append(f"--- {r.doc_uid}  {zc} {getattr(r, zc):+.2f}  z_sh_em3 {r.z_sh_em3:+.2f}  {r.n_tokens} tokens\n"
                         f"{' '.join(str(txt)[:700].split())}\n")
    (OUT / "samples.txt").write_text("\n".join(S))
    print(f"wrote {OUT / 'report.txt'} and {OUT / 'samples.txt'}")


if __name__ == "__main__":
    main()
