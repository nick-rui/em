"""Document lists for the new-recipe CPT corpora (em/cpt_recipe/PREREG.md): which pool documents to read from the
Dolma 3 shards, with their copy group. Runs on the GPU box (CPU only, a few GB of RAM).

It rebuilds the research pass's copy-group table with that pass's own code (research_copy/build_manifests.py, copied
unchanged from ~/code/em-olmo-research/research/cpt_manifests/scripts/ on 2026-11-13; sha256 in research_copy/SHA256SUMS),
so gid, the exclusions (labelled and held-out copy groups) and the take orders match the research manifests. Checks:
the rebuilt held-out set equals heldout_random.parquet, and every rand1b_s21 / top2b_slice row has the same gid here.

Outputs (--out):
  groups.parquet   one row per copy group: gid (= rank by z_sh_em3), z, copies, n_tokens (capped at 512), source,
                   topic, bucket, excl, in_{top1b,rand1b,blind1b} (sprint sets), rep_doc_uid (the group's first
                   document in pool file order: the copy a deduplicated arm uses)
  extract.parquet  the documents to read: doc_uid, gid, why (heldout / rand1b_s21 / top), sprint (also store the
                   sprint's 1,024-token cut). heldout_random and every rand1b_s21 row (all copies, for the old-recipe
                   arm) get both; the representative document of each of the first --top_groups eligible groups by z
                   gets whole-document tokens only.

  python em/cpt_recipe/whole_lists.py --pool data/pool20m --repo . --manifests data/manifests \
      --work data/cpt_recipe/work --out data/cpt_recipe/lists [--top_groups 2500000]
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "research_copy"))
import build_manifests as bm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--repo", required=True, help="dir holding data/probe_labels{,2}/labels*.parquet")
    ap.add_argument("--manifests", required=True, help="dir with rand1b_s21 / top2b_slice / heldout_random parquet")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_groups", type=int, default=2_500_000)
    a = ap.parse_args()
    work, out, man = Path(a.work), Path(a.out), Path(a.manifests)
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect(str(work / "pool.duckdb"))
    c.execute(f"SET memory_limit='64GB'; SET threads=16; SET enable_progress_bar=false; "
              f"SET temp_directory='{work / 'tmp'}'; SET preserve_insertion_order=true")
    bm.build_tables(c, str(Path(a.pool).resolve()), str(Path(a.repo).resolve()))
    held_chk = bm.heldout(c)
    g, est, n_incons = bm.group_arrays(c)
    bm.log(f"{len(g):,} groups, {int(g.copies.sum()):,} docs, excluded {int(g.excl.sum()):,}")
    rep = c.execute("SELECT gid, arg_min(doc_uid, rid) AS rep_doc_uid FROM P GROUP BY gid ORDER BY gid").df()
    assert (rep.gid.to_numpy() == np.arange(len(g))).all()
    g["rep_doc_uid"] = rep.rep_doc_uid.to_numpy()
    keep = ["gid", "z", "copies", "n_tokens", "source", "topic", "bucket", "excl", "in_top1b", "in_rand1b",
            "in_blind1b", "rep_doc_uid"]
    g[keep].to_parquet(out / "groups.parquet")

    # consistency with the research manifests
    held = c.execute("SELECT doc_uid, gid FROM HELD").df()
    ref = pd.read_parquet(man / "heldout_random.parquet")
    checks = dict(heldout_shared_groups_with_sprint_sets=held_chk,
                  heldout_same_docs=bool(set(held.doc_uid) == set(ref.doc_uid)), heldout_n=len(held))
    for name in ("rand1b_s21", "top2b_slice"):
        m = pd.read_parquet(man / f"{name}.parquet", columns=["doc_uid", "gid"])
        c.register("m", m)
        bad = c.execute("SELECT count(*) FROM m JOIN P USING (doc_uid) WHERE m.gid != P.gid").fetchone()[0]
        miss = c.execute("SELECT count(*) FROM m ANTI JOIN P USING (doc_uid)").fetchone()[0]
        c.unregister("m")
        checks[f"{name}_rows"] = len(m)
        checks[f"{name}_gid_mismatch"] = int(bad)
        checks[f"{name}_missing"] = int(miss)
    bm.log("checks", checks)
    assert checks["heldout_same_docs"] and checks["rand1b_s21_gid_mismatch"] == 0 and checks["top2b_slice_gid_mismatch"] == 0

    rand = pd.read_parquet(man / "rand1b_s21.parquet", columns=["doc_uid", "gid"])
    elig = g.loc[~g.excl, ["gid", "rep_doc_uid", "z"]].iloc[:a.top_groups]
    parts = [held.assign(why="heldout", sprint=True), rand.assign(why="rand1b_s21", sprint=True),
             elig.rename(columns={"rep_doc_uid": "doc_uid"})[["doc_uid", "gid"]].assign(why="top", sprint=False)]
    ex = pd.concat(parts, ignore_index=True)
    # a doc listed twice keeps sprint=True if any listing needs it
    ex = (ex.groupby("doc_uid", sort=False).agg(gid=("gid", "first"), why=("why", lambda s: "+".join(sorted(set(s)))),
                                                sprint=("sprint", "max")).reset_index())
    ex.to_parquet(out / "extract.parquet")
    meta = dict(checks=checks, n_extract=len(ex), n_sprint=int(ex.sprint.sum()), top_groups=a.top_groups,
                top_z_min=float(elig.z.min()), token_conflicts_within_group=n_incons,
                why_counts=ex.why.value_counts().to_dict())
    json.dump(meta, open(out / "lists_meta.json", "w"), indent=2, default=str)
    bm.log(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
