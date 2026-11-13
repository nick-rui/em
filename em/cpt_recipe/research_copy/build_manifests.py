"""CPT corpus manifests on the 20M pool (CPU only; claude-box). Document-ID lists with per-document token counts, for a
GPU box to tokenize later with materialize.py. Follows the rule of em/scripts/pool20m_cpt_data.py: highest z_sh_em3
first with all repeated copies, first 1,024 tokens + EOS per document, 1B tokens, shuffled.

  uv run --with duckdb --with pyarrow --with numpy --with pandas python build_manifests.py [--pool DIR] [--work DIR]

Writes ../*.parquet (one manifest per arm), ../manifests.json (budgets, seeds, per-arm stats) and ../overlaps.csv.

Units. A copy group is all pool documents with the same (shard, round(z_sh_em3, 5), round(z_sh_em3_perp_ben, 5),
n_tokens): pool_dups.py's key, also used by pool20m_cpt_data.rand() (checked by text there). The exact float key would
split 5,226 of these groups whose z differ by float noise (at most 2.1e-6). Every draw takes or leaves whole copy
groups, so a selected or random arm keeps the pool's duplication, as cpt_top1b and cpt_rand1b did.

Exclusions (every arm). Copy groups holding a labelled document (data/probe_labels/labels.parquet,
data/probe_labels2/labels2.parquet; 122k docs) or a held-out-loss document (heldout_random, 3,000 docs) are left out.
heldout_random/docs.parquet is not on claude-box or HF; it is rebuilt here with the exact pandas call of
pool20m_cpt_data.heldout() (RandomState(1).permutation over the non-TOP rows in file order, first 3,000) and checked
against cpt_rand1b / cpt_blind1b, which both left its copy groups out.

Token counts. cpt_tokens = min(len, 1024) + 1. n_tokens (scored, capped at 512) gives it exactly when n_tokens < 512
(cpt_tokens = n_tokens + 1; checked on 1.0M docs of the three existing sets). For n_tokens = 512 it is exact only for
documents in cpt_top1b / cpt_rand1b / cpt_blind1b (propagated to their copy group); otherwise it is estimated as the
mean cpt_tokens of n_tokens = 512 documents of the same source in cpt_rand1b (a random sample). Manifests carry
tok_exact; materialize.py recomputes every count from text and cuts each arm at its exact budget.

Manifest rows are in take order (column `order`). Each arm lists documents up to 1.06x its token budget on the
estimate; materialize.py takes rows in order until the exact budget (per `part`) and then shuffles with the arm seed.
"""
import argparse
import json
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
BUDGET, SEQ, BUF = 1_000_000_000, 1024, 1.06
SLICE = 2_000_000_000
SEEDS_TOP2B = [11, 12, 13]
SEEDS_RAND = [21, 22, 23]
SEED_MIX_SUB, SEED_MIX_RAND = 31, 32
SEED_MATCH = 41
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def parse_dir(d):   # em/prep_dolma.py
    m = re.match(r"(common_crawl|olmocr_science_pdfs)-(.+?)(?:-(\d{4}|part\d))?$", d)
    if m:
        return m.group(1), m.group(2), m.group(3) or ""
    return d.split("-")[0], d, ""


def connect(work):
    c = duckdb.connect(str(work / "pool.duckdb"))
    c.execute(f"SET memory_limit='1500MB'; SET threads=2; SET enable_progress_bar=false; "
              f"SET temp_directory='{work / 'tmp'}'; SET preserve_insertion_order=true")
    return c


def build_tables(c, pool, repo):
    """P: one row per pool doc (file row order) with its copy group gid; G: one row per copy group, gid = rank by
    z_sh_em3 descending (ties broken by shard, perp z, n_tokens)."""
    if "G" in {r[0] for r in c.execute("show tables").fetchall()}:
        log("tables exist; reusing")
        return
    log("loading pool")
    # key = pool_dups.py's: z and perp z rounded to 5 decimals (exact float equality splits true copies)
    c.execute(f"""CREATE TABLE P0 AS SELECT file_row_number AS rid, doc_uid, shard, z_sh_em3 AS z,
                  round(z_sh_em3, 5) AS zr, round(z_sh_em3_perp_ben, 5) AS zpr, n_tokens::INTEGER AS n_tokens
                  FROM read_parquet('{pool}/scores.parquet', file_row_number=true)""")
    c.execute("""CREATE TABLE G0 AS SELECT shard, zr, zpr, n_tokens, max(z) AS z, count(*)::INTEGER AS copies
                 FROM P0 GROUP BY shard, zr, zpr, n_tokens""")
    c.execute("""CREATE TABLE G1 AS SELECT row_number() OVER (ORDER BY z DESC, shard, zpr, n_tokens)::INTEGER - 1 AS gid,
                 * FROM G0""")
    c.execute("DROP TABLE G0")
    log("joining docs to groups")
    c.execute("""CREATE TABLE P AS SELECT P0.rid, P0.doc_uid, P0.z AS dz, G1.gid FROM P0
                 JOIN G1 USING (shard, zr, zpr, n_tokens) ORDER BY P0.rid""")
    c.execute("DROP TABLE P0")
    # exact token counts from the three tokenized sets
    sets = {k: f"{pool}/cpt_{k}/docs.parquet" for k in ("top1b", "rand1b", "blind1b")}
    u = " UNION ALL ".join(f"SELECT doc_uid, cpt_tokens::INTEGER AS t FROM '{p}'" for p in sets.values())
    c.execute(f"""CREATE TABLE K AS SELECT P.gid, min(t) AS tmin, max(t) AS tmax FROM ({u}) x JOIN P USING (doc_uid)
                  GROUP BY P.gid""")
    for k, p in sets.items():
        c.execute(f"CREATE TABLE S_{k} AS SELECT DISTINCT P.gid FROM '{p}' x JOIN P USING (doc_uid)")
        c.execute(f"CREATE TABLE D_{k} AS SELECT doc_uid, cpt_tokens::INTEGER AS cpt_tokens FROM '{p}'")
    lab = " UNION ".join(f"SELECT doc_uid FROM '{repo}/data/{f}'" for f in
                         ("probe_labels/labels.parquet", "probe_labels2/labels2.parquet"))
    c.execute(f"CREATE TABLE LAB AS SELECT DISTINCT P.gid FROM ({lab}) l JOIN P USING (doc_uid)")
    c.execute("""CREATE TABLE G AS SELECT G1.*, K.tmin, K.tmax FROM G1 LEFT JOIN K USING (gid) ORDER BY gid""")
    c.execute("DROP TABLE G1; DROP TABLE K")
    c.execute("CHECKPOINT")
    log("tables built")


def heldout(c):
    """Rebuild heldout_random: pandas s[~s.doc_uid.isin(top)].sample(3000, random_state=1) == RandomState(1).choice(n,
    3000, replace=False) == RandomState(1).permutation(n)[:3000] on the non-TOP rows in file order."""
    rid = c.execute("""SELECT P.rid FROM P ANTI JOIN D_top1b t USING (doc_uid) ORDER BY P.rid""").fetchnumpy()["rid"]
    pick = rid[np.random.RandomState(1).permutation(len(rid))[:3000]]
    del rid
    c.register("hp", pd.DataFrame({"rid": pick}))
    c.execute("CREATE OR REPLACE TABLE HELD AS SELECT P.rid, P.doc_uid, P.gid FROM hp JOIN P USING (rid)")
    chk = {k: c.execute(f"SELECT count(*) FROM HELD JOIN S_{k} USING (gid)").fetchone()[0]
           for k in ("top1b", "rand1b", "blind1b")}
    log("held-out rebuilt; copy groups shared with", chk)
    return chk


def group_arrays(c):
    c.execute("""CREATE OR REPLACE TABLE DIRS AS SELECT dir, (row_number() OVER (ORDER BY dir) - 1)::SMALLINT AS did
                 FROM (SELECT DISTINCT split_part(shard, '__', 1) AS dir FROM G)""")
    dirs = c.execute("SELECT dir, did FROM DIRS ORDER BY did").df()
    g = c.execute("""SELECT gid, z, n_tokens, copies, did, tmin, tmax FROM G
                     JOIN DIRS ON split_part(G.shard, '__', 1) = DIRS.dir ORDER BY gid""").df()
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    did = g.pop("did").to_numpy()
    for j, col in enumerate(("source", "topic", "bucket")):
        lab = pd.Categorical([parse_dir(x)[j] for x in dirs.dir])
        g[col] = pd.Categorical.from_codes(lab.codes[did], lab.categories)
    stb = pd.Categorical(dirs.dir.map(lambda x: "|".join(parse_dir(x))))
    g["stb"] = pd.Categorical.from_codes(stb.codes[did], stb.categories)
    excl = set(c.execute("SELECT gid FROM LAB UNION SELECT gid FROM HELD").fetchnumpy()["gid"].tolist())
    g["excl"] = g.gid.isin(excl)
    for k in ("top1b", "rand1b", "blind1b"):
        g[f"in_{k}"] = g.gid.isin(set(c.execute(f"SELECT gid FROM S_{k}").fetchnumpy()["gid"].tolist()))
    # per-copy token count
    short = g.n_tokens < 512
    known = g.tmin.notna()
    lng = g[known & ~short & g.in_rand1b]
    est = (lng.tmin).groupby(lng.source, observed=True).mean()
    g["tok"] = np.where(short, g.n_tokens + 1, np.where(known, g.tmin, g.source.map(est).astype(float)))
    g["tok_exact"] = short | known
    g["gtok"] = g.tok * g.copies
    return g, est, int((g.tmin != g.tmax).sum())


def take_until(order_gids, gtok, budget, buf=BUF):
    """Prefix of order_gids whose estimated tokens reach buf x budget; also the prefix that reaches budget."""
    cs = np.cumsum(gtok[order_gids])
    k_buf = int(np.searchsorted(cs, buf * budget)) + 1
    k_bud = int(np.searchsorted(cs, budget)) + 1
    return order_gids[:k_buf], k_bud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(Path.home() / "code/em-olmo/data/pool20m"))
    ap.add_argument("--repo", default=str(Path.home() / "code/em-olmo"))
    ap.add_argument("--work", required=True, help="scratch dir for the duckdb file (about 2 GB)")
    a = ap.parse_args()
    work = Path(a.work)
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    c = connect(work)
    build_tables(c, a.pool, a.repo)
    held_chk = heldout(c)
    g, est, n_incons = group_arrays(c)
    log(f"{len(g):,} copy groups; {int(g.copies.sum()):,} docs; excluded groups {int(g.excl.sum()):,} "
        f"({int(g.copies[g.excl].sum()):,} docs); token-count conflicts within a group {n_incons}")
    gtok = g.gtok.to_numpy()
    elig = ~g.excl.to_numpy()
    rank = np.flatnonzero(elig)   # eligible gids in z order (gid is the z rank)
    arms, R = {}, {}
    # ---- (A) top-2B slice and its draws
    cs = np.cumsum(gtok[rank])
    k2 = int(np.searchsorted(cs, SLICE)) + 1
    slice_g = rank[:k2]
    k1 = int(np.searchsorted(cs, BUDGET)) + 1
    top1_g = rank[:k1]   # TOP 1B on this rule (labelled / held-out groups left out)
    R["slice"] = dict(n_groups=k2, n_docs=int(g.copies.to_numpy()[slice_g].sum()), tokens_est=float(cs[k2 - 1]),
                      tokens_exact_share=float((gtok[slice_g] * g.tok_exact.to_numpy()[slice_g]).sum() / cs[k2 - 1]),
                      z_min=float(g.z.to_numpy()[slice_g].min()))
    arms["top2b_slice"] = dict(gids=slice_g, parts={"slice": (slice_g, None)}, seed=None, kind="the top-2B-token slice "
                               "(reference list; not a training arm)")
    arms["top1b_rule"] = dict(gids=None, parts={"top": take_until(rank, gtok, BUDGET)}, seed=0,
                              kind="highest z first with copies until 1B (cpt_top1b's rule, labelled/held-out left out)")
    for s in SEEDS_TOP2B:
        o = np.random.default_rng(s).permutation(slice_g)
        arms[f"top2b_draw_s{s}"] = dict(parts={"top2b": take_until(o, gtok, BUDGET)}, seed=s,
                                        kind="random copy groups of the top-2B slice until 1B")
    pool_order = {}
    for s in SEEDS_RAND:
        o = np.random.default_rng(s).permutation(rank)
        pool_order[s] = o
        arms[f"rand1b_s{s}"] = dict(parts={"random": take_until(o, gtok, BUDGET)}, seed=s,
                                    kind="random copy groups of the pool until 1B")
    # ---- (B) mix ratios. Selected part: TOP 1B groups (top1_g). Random part: pool groups outside top1_g, one seeded
    # order shared by all ratios (so the random parts are nested).
    in_top1 = np.zeros(len(g), bool)
    in_top1[top1_g] = True
    ro = np.random.default_rng(SEED_MIX_RAND).permutation(rank)
    ro = ro[~in_top1[ro]]
    sub = np.random.default_rng(SEED_MIX_SUB).permutation(top1_g)
    for p in (0, 25, 50, 100):
        bs, br = BUDGET * p // 100, BUDGET * (100 - p) // 100
        parts = {}
        if bs:
            parts["selected"] = take_until(sub, gtok, bs) if p < 100 else take_until(rank, gtok, bs)
        if br:
            parts["random"] = take_until(ro, gtok, br)
        arms[f"mix{p:03d}_sub"] = dict(parts=parts, seed=SEED_MIX_SUB * 1000 + p, kind=(
            f"{p}% selected tokens (random copy groups of TOP 1B), {100 - p}% random pool groups outside TOP 1B"))
    for p in (25, 50):
        bs, br = BUDGET * p // 100, BUDGET * (100 - p) // 100
        arms[f"mix{p:03d}_head"] = dict(parts={"selected": take_until(rank, gtok, bs),
                                               "random": take_until(ro, gtok, br)}, seed=SEED_MIX_SUB * 1000 + p + 1,
                                        kind=f"{p}% selected tokens (highest z first), {100 - p}% random outside TOP 1B")
    # ---- (C) matched random control
    match_info = matched(g, gtok, rank, slice_g, top1_g, pool_order[SEEDS_RAND[0]], arms)
    # ---- write manifests
    write_all(c, g, arms, R, held_chk, est, match_info)


def cells(g, lb):
    """Pool-wide matching cell: source x topic x bucket x length bin (n_tokens; 512 = capped) x copies bin."""
    cb = np.select([g.copies == 1, g.copies <= 3, g.copies <= 10], [0, 1, 2], 3)
    nb = np.digitize(g.n_tokens.to_numpy(), lb)
    return g.stb.cat.codes.to_numpy().astype(np.int64) * 100 + nb * 10 + cb


def cell_name(g, code):
    return f"{g.stb.cat.categories[code // 100]}|L{code % 100 // 10}|C{code % 10}"


LEN_BINS = [64, 128, 192, 256, 320, 384, 448, 512]   # n_tokens bins; the last bin (== 512) is every doc of 512+ tokens


def matched(g, gtok, rank, slice_g, top1_g, rand_order, arms):
    """Random groups outside the top-2B slice, token shares matched to TOP 1B per cell. Within each cell the groups are
    taken in a seeded random order until the cell's TOP token share x 1.06B; a cell short of eligible tokens is
    reported (its gap is not refilled from other cells)."""
    code = cells(g, LEN_BINS)
    out_slice = np.ones(len(g), bool)
    out_slice[slice_g] = False
    cand = rank[out_slice[rank]]
    cand = np.random.default_rng(SEED_MATCH).permutation(cand)
    target = pd.Series(gtok[top1_g]).groupby(code[top1_g]).sum()
    target = target / target.sum()
    cc = pd.DataFrame({"gid": cand, "c": code[cand], "t": gtok[cand]})
    cc["cs"] = cc.groupby("c").t.cumsum()
    tgt = cc.c.map(target).fillna(0).to_numpy() * BUF * BUDGET
    keep = (cc.cs.to_numpy() - cc.t.to_numpy()) < tgt   # groups until the cell's target is reached (last one crosses)
    sel = cc[keep].assign(prog=(cc.cs.to_numpy() / np.maximum(tgt, 1))[keep])
    # take order: by progress through the cell's target, so a cut at 1B trims every cell by the same share (the plain
    # candidate order would front-load cells with many eligible groups and under-fill scarce cells)
    sel = sel.sort_values("prog", kind="stable")
    got = sel.groupby("c").t.sum()
    need = target * BUF * BUDGET
    short = (need - got.reindex(need.index).fillna(0)).clip(lower=0)
    short = short[short > need * 0.01]
    sel_g = sel.gid.to_numpy()
    arms["matched1b"] = dict(parts={"matched": (sel_g, int(np.searchsorted(np.cumsum(gtok[sel_g]), BUDGET)) + 1)},
                             seed=SEED_MATCH, kind="random groups outside the top-2B slice, matched to TOP 1B on "
                             "source x topic x quality bucket x length bin x copies bin (token shares per cell)")
    return dict(n_cells_top=int(len(target)), short_cells={cell_name(g, int(k)): float(v / 1e6)
                                                           for k, v in short.items()},
                short_tokens_M=float(short.sum() / 1e6), code=code)


def write_all(c, g, arms, R, held_chk, est, mi):
    z, cop, tok, tex = (g[k].to_numpy() for k in ("z", "copies", "tok", "tok_exact"))
    gtok = g.gtok.to_numpy()
    meta, lines = {}, []
    docsets = {}
    for name, a in arms.items():
        frames = []
        for part, (gids, k_bud) in a["parts"].items():
            frames.append(pd.DataFrame({"gid": gids, "part": part, "grp_order": np.arange(len(gids)),
                                        "in_budget_est": np.arange(len(gids)) < (k_bud if k_bud else len(gids))}))
        f = pd.concat(frames, ignore_index=True)
        c.register("f", f)
        d = c.execute("""SELECT P.doc_uid, P.dz, f.gid, f.part, f.grp_order, f.in_budget_est FROM f JOIN P USING (gid)
                         ORDER BY f.part, f.grp_order, P.rid""").df()
        c.unregister("f")
        d["shard"] = d.doc_uid.str.split(":").str[0].str.replace(".jsonl.zst", "", regex=False).str.replace(
            ".jsonl.gz", "", regex=False).str.replace("/", "__", regex=False)
        d["z_sh_em3"] = d.pop("dz").astype(np.float32)   # the doc's own z (copies differ by < 3e-6)
        d["n_tokens"] = g.n_tokens.to_numpy()[d.gid].astype(np.int16)
        d["cpt_tokens"] = tok[d.gid].round().astype(np.int16)
        d["tok_exact"] = tex[d.gid]
        d["copies"] = cop[d.gid].astype(np.int32)
        d["order"] = np.arange(len(d), dtype=np.int32)
        d["arm"] = name
        d = d[["doc_uid", "shard", "arm", "part", "order", "gid", "copies", "z_sh_em3", "n_tokens", "cpt_tokens",
               "tok_exact", "in_budget_est"]]
        path = OUT / f"{name}.parquet"
        d.to_parquet(path, compression="zstd", index=False)
        b = d.loc[d.in_budget_est, ["gid", "part", "z_sh_em3", "cpt_tokens", "tok_exact"]].reset_index(drop=True)
        docsets[name] = b
        del d
        m = dict(kind=a["kind"], seed=a["seed"], file=path.name, size_MB=round(path.stat().st_size / 1e6, 1),
                 rows=int(pd.read_parquet(path, columns=["gid"]).shape[0]), parts={})
        for part, (gids, k_bud) in a["parts"].items():
            bb = b[b.part == part]
            m["parts"][part] = dict(
                budget_tokens=part_budget(name, part), docs_in_budget=len(bb), groups_in_budget=int(bb.gid.nunique()),
                tokens_est=int(bb.cpt_tokens.astype(np.int64).sum()),
                exact_token_share=float((bb.cpt_tokens.astype(np.int64) * bb.tok_exact).sum()
                                        / max(1, bb.cpt_tokens.astype(np.int64).sum())),
                z_mean_docs=float(bb.z_sh_em3.mean()),
                z_mean_tokens=float(np.average(bb.z_sh_em3, weights=bb.cpt_tokens)))
        m["z_mean_docs"] = float(b.z_sh_em3.mean())
        m["z_mean_tokens"] = float(np.average(b.z_sh_em3, weights=b.cpt_tokens))
        m["docs_in_budget"] = len(b)
        m["unique_share_docs"] = float(b.gid.nunique() / len(b))
        meta[name] = m
        log(name, m["rows"], "rows", m["size_MB"], "MB", "z", round(m["z_mean_docs"], 3))
    # originals
    orig = {}
    for k in ("top1b", "rand1b", "blind1b"):
        orig[k] = c.execute(f"SELECT P.gid, D.cpt_tokens FROM D_{k} D JOIN P USING (doc_uid)").df()
    ov = overlaps(docsets, orig, z)
    bal = balance(c, g, docsets, mi)
    rep = dict(slice=R["slice"], heldout_rebuild_shared_groups=held_chk, token_estimate_by_source=est.round(1).to_dict(),
               arms=meta, overlaps=ov, matched=dict(n_cells_top=mi["n_cells_top"], short_tokens_M=mi["short_tokens_M"],
                                                    short_cells=mi["short_cells"]), balance=bal)
    json.dump(rep, open(OUT / "manifests.json", "w"), indent=1, default=float)
    log("wrote manifests.json")


def part_budget(name, part):
    if name == "top2b_slice":
        return None
    if name.startswith("mix"):
        p = int(name[3:6])
        return BUDGET * p // 100 if part == "selected" else BUDGET * (100 - p) // 100
    return BUDGET


def overlaps(docsets, orig, z):
    """Overlap of in-budget document sets (by doc, by copy group, by tokens)."""
    sets = {k: v for k, v in docsets.items() if k != "top2b_slice"}
    named = {**{f"orig_{k}": dict(g=set(v.gid), d=v) for k, v in orig.items()},
             **{k: dict(g=set(v.gid), d=v) for k, v in sets.items()}}
    rows = []
    keys = [k for k in named if k.startswith(("top2b_draw", "rand1b_s"))] + ["orig_top1b", "orig_rand1b",
                                                                             "orig_blind1b", "top1b_rule", "matched1b"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ga, gb = named[a]["g"], named[b]["g"]
            da = named[a]["d"]
            tok_col = "cpt_tokens"
            sh = da[da.gid.isin(gb)]
            rows.append(dict(a=a, b=b, groups_a=len(ga), groups_b=len(gb), shared_groups=len(ga & gb),
                             share_groups_of_a=len(ga & gb) / len(ga),
                             share_tokens_of_a=float(sh[tok_col].astype(np.int64).sum()
                                                     / da[tok_col].astype(np.int64).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "overlaps.csv", index=False)
    return df.to_dict("records")


def balance(c, g, docsets, mi):
    """Feature shares (tokens) for TOP 1B (top1b_rule), matched1b and random (rand1b_s21), plus z means."""
    res = {}
    cb = np.select([g.copies == 1, g.copies <= 3, g.copies <= 10], [0, 1, 2], 3)
    cols = dict(source=(g.source.cat.codes.to_numpy(), list(g.source.cat.categories)),
                topic=(g.topic.cat.codes.to_numpy(), list(g.topic.cat.categories)),
                bucket=(g.bucket.cat.codes.to_numpy(), list(g.bucket.cat.categories)),
                len_bin=(np.digitize(g.n_tokens.to_numpy(), LEN_BINS),
                         ["<64", "64-127", "128-191", "192-255", "256-319", "320-383", "384-447", "448-511", "512+"]),
                copies_bin=(cb, ["1", "2-3", "4-10", "11+"]))
    arms = dict(TOP="top1b_rule", matched="matched1b", random="rand1b_s21")
    for feat, (arr, names) in cols.items():
        t = {}
        for lab, name in arms.items():
            b = docsets[name]
            w = b.cpt_tokens.astype(np.int64)
            t[lab] = pd.Series(w.to_numpy()).groupby(arr[b.gid.to_numpy()]).sum() / w.sum()
        tb = pd.DataFrame(t).fillna(0)
        tb.index = [names[i] or "(none)" for i in tb.index]
        tb = tb.sort_values("TOP", ascending=False)
        res[feat] = dict(table=tb.round(4).to_dict("index"),
                         tvd_matched_top=float(0.5 * (tb.matched - tb.TOP).abs().sum()),
                         tvd_random_top=float(0.5 * (tb.random - tb.TOP).abs().sum()))
    # z gap
    zs = {lab: dict(docs=float(docsets[n].z_sh_em3.mean()),
                    tokens=float(np.average(docsets[n].z_sh_em3, weights=docsets[n].cpt_tokens)))
          for lab, n in arms.items()}
    zs["gap_remaining_docs"] = (zs["TOP"]["docs"] - zs["matched"]["docs"]) / (zs["TOP"]["docs"] - zs["random"]["docs"])
    zs["gap_remaining_tokens"] = ((zs["TOP"]["tokens"] - zs["matched"]["tokens"])
                                  / (zs["TOP"]["tokens"] - zs["random"]["tokens"]))
    res["z"] = zs
    return res


if __name__ == "__main__":
    main()
