"""Numbers for the open data decisions in em/cpt_recipe/PREREG.md section 6, from the extract cache (whole-document
token counts) and the copy-group table. CPU only, on the box.

1. Token rule: for 1B TOP tokens, highest z first over eligible copy groups, under four rules
   sprint (first 1,024 ids + EOS, every copy) | cut1024_dedup (one copy) | whole_copies | whole_dedup (new rule):
   groups, documents, the z reached, mean z by documents and by tokens, overlap with the sprint's cpt_top1b groups.
   The sprint cut is approximated as min(n_whole - 1, 1024) + 1 for TOP groups (their sprint ids were not extracted;
   exact for rand1b_s21, where the approximation is checked against the exact counts).
2. TOP run designs under the new rule: strict top 1B (3 runs share every document), random halves of the top 2B
   (as top2b_draw_s11-13), disjoint thirds of the top 3B: z reached, mean z of each draw, pairwise token overlap.
3. RANDOM weighting over rand1b_s21's groups (whole documents, one copy): group-uniform (V-new's take order) vs
   weighting each group by its copies (OLMo's own token stream holds an upsampled document once per copy): token share
   by source, next to the stage-1 mix's shares (research/cpt_audit/recipe_audit.md [S4]).
4. TOP tokens past positions 512 and 1,024 of their document (unscored by the selection score).

  python em/cpt_recipe/data_decisions.py > results/cpt_recipe/data_decisions.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LISTS, CACHE = ROOT / "data/cpt_recipe/lists", ROOT / "data/cpt_recipe/cache"
B = 1_000_000_000
# stage-1 token shares by source from OLMo-mix-0625-official (audit [S4]); the pool's source names
MIX = {"common_crawl": 0.761, "olmocr_science_pdfs": 0.136, "stack_edu": 0.069, "finemath": 0.026, "rpj": 0.0086,
       "dolma1_7": 0.0004}


def cache_lengths():
    rows = []
    for f in sorted(CACHE.glob("*.parquet")):
        rows.append(pd.read_parquet(f, columns=["doc_uid", "n_whole", "n_sprint"]))
    return pd.concat(rows, ignore_index=True).set_index("doc_uid")


def take(order_tokens, budget):
    """Rows of a prefix that reaches budget (the crossing row kept)."""
    return int(np.searchsorted(np.cumsum(order_tokens), budget)) + 1


def main():
    g = pd.read_parquet(LISTS / "groups.parquet")
    ex = pd.read_parquet(LISTS / "extract.parquet")
    n = cache_lengths()
    print(f"extract cache: {len(n):,} documents of {len(ex):,} listed")
    out = {}

    # ---- whole-document lengths of the listed documents
    top = g[~g.excl].copy()
    top["n_whole"] = top.rep_doc_uid.map(n.n_whole)
    have = top.n_whole.notna()
    k_have = int(np.argmin(have.to_numpy())) if not have.all() else len(top)
    top = top.iloc[:k_have]   # the z-ordered prefix with whole-document counts (the extracted top groups)
    top["w"] = top.n_whole.astype(np.int64)
    top["cut"] = np.minimum(top.w - 1, 1024) + 1
    print(f"TOP groups with whole counts: {len(top):,} (z down to {top.z.min():.3f})")

    # check the cut approximation on rand1b_s21 (exact sprint counts there)
    rr = ex[ex.why.str.contains("rand1b_s21")].copy()
    rr["n_whole"] = rr.doc_uid.map(n.n_whole)
    rr["n_sprint"] = rr.doc_uid.map(n.n_sprint)
    rr = rr.dropna()
    approx = np.minimum(rr.n_whole - 1, 1024) + 1
    out["cut_approx_check"] = dict(docs=len(rr), exact_tokens=int(rr.n_sprint.sum()), approx_tokens=int(approx.sum()),
                                   ratio=float(approx.sum() / rr.n_sprint.sum()),
                                   docs_differing=int((approx != rr.n_sprint).sum()))

    # ---- 1. token rules, strict top cut to 1B
    rules = {"sprint (cut 1,024 + EOS, every copy)": top.cut * top.copies,
             "cut1024_dedup (one copy)": top.cut,
             "whole_copies": top.w * top.copies,
             "whole_dedup (new rule)": top.w}
    out["rules"] = {}
    for name, tok in rules.items():
        tok = tok.to_numpy()
        k = take(tok, B)
        if k > len(top):
            out["rules"][name] = "not reached within the extracted groups"
            continue
        s = top.iloc[:k]
        docs = s.copies if "copies" in name or "every copy" in name else pd.Series(1, index=s.index)
        out["rules"][name] = dict(
            groups=k, docs=int(docs.sum()), tokens=int(tok[:k].sum()), z_reached=round(float(s.z.iloc[-1]), 3),
            z_mean_docs=round(float((s.z * docs).sum() / docs.sum()), 3),
            z_mean_tokens=round(float((s.z * tok[:k]).sum() / tok[:k].sum()), 3),
            share_groups_in_sprint_top1b=round(float(s.in_top1b.mean()), 3),
            unique_tokens=int(s.w.sum() if "whole" in name else s.cut.sum()))

    # ---- 2. TOP run designs under the new rule
    w = top.w.to_numpy()
    designs = {}
    for label, slice_b, n_draws, mode in (("strict top 1B", 1, 3, "same"), ("random halves of top 2B", 2, 3, "rand"),
                                          ("disjoint thirds of top 3B", 3, 3, "disjoint")):
        k = take(w, slice_b * B)
        if k > len(top):
            designs[label] = "slice not reached within the extracted groups"
            continue
        sl = top.iloc[:k]
        draws = []
        if mode == "same":
            draws = [np.arange(take(w, B))] * n_draws
        elif mode == "rand":
            for s in (11, 12, 13):
                o = np.random.default_rng(s).permutation(k)
                draws.append(np.sort(o[:take(w[o], B)]))
        else:
            o = np.random.default_rng(14).permutation(k)
            cs = np.cumsum(w[o])
            cuts = [0] + [int(np.searchsorted(cs, (i + 1) * cs[-1] / 3)) + 1 for i in range(3)]
            draws = [np.sort(o[cuts[i]:cuts[i + 1]]) for i in range(3)]
        stats = []
        for d in draws:
            dd = sl.iloc[d]
            stats.append(dict(groups=len(d), tokens=int(dd.w.sum()), z_mean_docs=round(float(dd.z.mean()), 3),
                              z_mean_tokens=round(float((dd.z * dd.w).sum() / dd.w.sum()), 3),
                              z_min=round(float(dd.z.min()), 3)))
        ov = []
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                a, b = set(draws[i]), set(draws[j])
                ov.append(round(float(sl.iloc[sorted(a & b)].w.sum() / sl.iloc[draws[i]].w.sum()), 3))
        designs[label] = dict(slice_groups=k, slice_z_reached=round(float(sl.z.iloc[-1]), 3), draws=stats,
                              pairwise_token_overlap=ov)
    out["top_designs"] = designs

    # ---- 4. TOP tokens past positions 512 / 1,024 (new rule, strict 1B)
    s = top.iloc[:take(w, B)]
    out["top_tokens_past"] = {str(p): round(float(np.maximum(s.w - p, 0).sum() / s.w.sum()), 3) for p in (512, 1024, 8192)}
    out["top_doc_length"] = dict(mean=round(float(s.w.mean()), 1), median=float(s.w.median()), p99=float(s.w.quantile(.99)),
                                 max=int(s.w.max()))

    # ---- 3. RANDOM weighting over rand1b_s21's groups (whole, one copy)
    man = pd.read_parquet(ROOT / "data/manifests/rand1b_s21.parquet", columns=["doc_uid", "gid", "order"])
    man = man.sort_values("order").drop_duplicates("gid")
    man["w"] = man.doc_uid.map(n.n_whole)
    man = man.dropna().merge(g[["gid", "copies", "source"]], on="gid")
    k200 = take(man.w.to_numpy(), 200_000_000)
    v = man.iloc[:k200]
    tok_uniform = (man.groupby("source", observed=True).w.sum() / man.w.sum())
    tok_copies = ((man.w * man.copies).groupby(man.source, observed=True).sum() / (man.w * man.copies).sum())
    tok_vnew = v.groupby("source", observed=True).w.sum() / v.w.sum()
    tab = pd.DataFrame({"stage1_mix": pd.Series(MIX), "vnew_first_200M": tok_vnew, "group_uniform_all": tok_uniform,
                        "copies_weighted_all": tok_copies}).fillna(0).round(4)
    out["random_weighting"] = dict(groups=len(man), whole_tokens=int(man.w.sum()), vnew_groups=k200,
                                   vnew_tokens=int(v.w.sum()), vnew_mean_doc=round(float(v.w.mean()), 1),
                                   copies_mean=round(float(man.copies.mean()), 3),
                                   token_share_by_source=tab.to_dict())
    print(json.dumps(out, indent=1, default=str))
    print("\nRANDOM token share by source (stage-1 mix from the audit):")
    print(tab.to_string())
    (ROOT / "results/cpt_recipe").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results/cpt_recipe/data_decisions.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
