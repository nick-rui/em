"""Write the heldout_random document list (3,000 docs) as a file, so the GPU-box extract pass has an input.

Same rule as build_manifests.heldout() and em/scripts/pool20m_cpt_data.heldout(): the non-TOP pool rows (TOP =
cpt_top1b/docs.parquet) in scores.parquet file order, then RandomState(1).permutation(n)[:3000]. Also checks, by
copy group (pool_dups.py key: shard, round(z, 5), round(perp z, 5), n_tokens), that no held-out group is in
cpt_top1b / cpt_rand1b / cpt_blind1b. build_manifests.py reported 0 / 0 / 0 for the same check.

  uv run --with duckdb --with pyarrow --with numpy --with pandas python heldout_random.py [--pool DIR]

Writes ../heldout_random.parquet (doc_uid, shard, rid, z_sh_em3, n_tokens, order). Usable by materialize.py extract
(it reads only doc_uid). Not an arm in manifests.json, so materialize.py assemble does not apply to it.
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(Path.home() / "code/em-olmo/data/pool20m"))
    a = ap.parse_args()
    c = duckdb.connect()
    c.execute("SET memory_limit='1200MB'; SET threads=2; SET preserve_insertion_order=true")
    sc = f"read_parquet('{a.pool}/scores.parquet', file_row_number=true)"
    top = f"'{a.pool}/cpt_top1b/docs.parquet'"
    rid = c.execute(f"""SELECT s.file_row_number AS rid FROM {sc} s
                        ANTI JOIN (SELECT DISTINCT doc_uid FROM {top}) t USING (doc_uid)
                        ORDER BY rid""").fetchnumpy()["rid"]
    n_nontop = len(rid)
    pick = rid[np.random.RandomState(1).permutation(len(rid))[:3000]]
    del rid
    c.register("hp", pd.DataFrame({"rid": pick, "order": np.arange(3000, dtype=np.int32)}))
    held = c.execute(f"""SELECT s.doc_uid, s.shard, s.file_row_number AS rid, s.z_sh_em3, s.n_tokens::SMALLINT AS n_tokens,
                         hp."order", round(s.z_sh_em3, 5) AS zr, round(s.z_sh_em3_perp_ben, 5) AS zpr
                         FROM {sc} s JOIN hp ON s.file_row_number = hp.rid ORDER BY hp."order" """).df()
    assert len(held) == 3000 and held.doc_uid.is_unique
    c.register("hk", held[["shard", "zr", "zpr", "n_tokens"]].drop_duplicates())
    grp = c.execute(f"""SELECT s.doc_uid FROM {sc} s JOIN hk ON s.shard = hk.shard
                        AND round(s.z_sh_em3, 5) = hk.zr AND round(s.z_sh_em3_perp_ben, 5) = hk.zpr
                        AND s.n_tokens = hk.n_tokens""").df()
    c.register("hg", grp)
    chk = {}
    for k in ("top1b", "rand1b", "blind1b"):
        p = f"'{a.pool}/cpt_{k}/docs.parquet'"
        chk[k] = dict(docs=int(c.execute(f"SELECT count(*) FROM hg JOIN (SELECT DISTINCT doc_uid FROM {p}) x USING (doc_uid)").fetchone()[0]))
    held.drop(columns=["zr", "zpr"]).to_parquet(OUT / "heldout_random.parquet", compression="zstd", index=False)
    rep = dict(n_nontop_rows=int(n_nontop), n_docs=3000, n_copy_groups=int(len(c.execute("SELECT * FROM hk").df())),
               n_docs_in_those_groups=int(len(grp)), docs_of_heldout_groups_in_sets=chk,
               n_tokens_capped512_sum=int(held.n_tokens.sum()), n_tokens_eq512=int((held.n_tokens == 512).sum()))
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(OUT / "heldout_random.json", "w"), indent=1)


if __name__ == "__main__":
    main()
