"""E2's RANDOM documents (PREREG.md section 6, item 3: copies-weighted, as V-new; Nick, 2026-11-13).

A fresh draw, not rand1b_s21: rand1b_s21's 683,192 groups hold only 1.10B whole-document tokens, so a 1B
copies-weighted draw from it would take nearly all of them and lose the weighting. Here every eligible pool document
(copy groups with a labelled or held-out document left out, as in every arm) is put in a seeded random order, and each
copy group enters at its first document in that order. A group with c copies is c times as likely to come early, which
is how OLMo's own token stream weights it (it holds an upsampled document once per copy). The group's representative
document (its first in pool file order, as every arm uses) is the copy that gets tokenized. The first --groups groups
are listed (about 1.6B whole-document tokens at V-new's 1,594 per document, so the 1B cut uses about 60%).

Uses the copy-group tables whole_lists.py built (data/cpt_recipe/work/pool.duckdb, lists/groups.parquet).
  python em/cpt_recipe/rand_cw_list.py --seed 24 --groups 1000000
    -> data/cpt_recipe/lists/rand_cw.parquet (doc_uid, gid, order), lists/rand_cw_extract.parquet (doc_uid, sprint)
  E2 extension (new RANDOM draws): --seed 25 --name rand_cw_d25 -> lists/rand_cw_d25.parquet, ...
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--groups", type=int, default=1_000_000)
    ap.add_argument("--work", default=str(ROOT / "data/cpt_recipe/work"))
    ap.add_argument("--lists", default=str(ROOT / "data/cpt_recipe/lists"))
    ap.add_argument("--name", default="rand_cw", help="output name: <name>.parquet, <name>_extract.parquet, "
                                                       "<name>_meta.json (default: E2 RANDOM run 1's draw, seed 24)")
    a = ap.parse_args()
    c = duckdb.connect(str(Path(a.work) / "pool.duckdb"), read_only=True)
    p = c.execute("SELECT rid, gid FROM P ORDER BY rid").fetchnumpy()
    g = pd.read_parquet(Path(a.lists) / "groups.parquet", columns=["gid", "excl", "rep_doc_uid", "copies", "z"])
    excl = g.excl.to_numpy()
    gid = p["gid"][~excl[p["gid"]]]                      # documents of eligible groups, in pool file order
    order = np.random.default_rng(a.seed).permutation(len(gid))
    first = pd.unique(gid[order])[:a.groups]             # groups by their first document in the random order
    sel = g.iloc[first][["gid", "rep_doc_uid", "copies", "z"]].rename(columns={"rep_doc_uid": "doc_uid"})
    sel["order"] = np.arange(len(sel))
    out = Path(a.lists)
    sel[["doc_uid", "gid", "order"]].to_parquet(out / f"{a.name}.parquet")
    sel[["doc_uid"]].assign(sprint=False).to_parquet(out / f"{a.name}_extract.parquet")
    elig = g[~g.excl]
    meta = dict(seed=a.seed, groups=len(sel), eligible_groups=int(len(elig)), eligible_docs=int(len(gid)),
                mean_copies_selected=float(sel.copies.mean()), mean_copies_eligible_groups=float(elig.copies.mean()),
                mean_copies_doc_weighted=float((elig.copies ** 2).sum() / elig.copies.sum()),
                z_mean_selected=float(sel.z.mean()))
    json.dump(meta, open(out / f"{a.name}_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
