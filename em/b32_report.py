"""Phases GN32 + PT32 report (pre-registered in TODO_32b.md): OLMo-3-32B stage 1, G and N at layer L*. CPU only.
Inputs (pulled from the box by scripts/judge_b32.sh and scripts/pull_b32.sh): judged responses, runs/b32_*,
results/b32/{variance_L*.json, ablate_L*/*.parquet, docs_proj_L*.parquet}.
Behaviour: rates as in nar_report.py. Useful: token-weighted mean Δloss (ablated − base), document bootstrap.
  python em/b32_report.py --layer L > results/b32/report_L.txt
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from r_report import ROOT, counts, fisher, fmt, newcombe

R = ROOT / "results" / "b32"
DOM = {"med": ("medical_questions", "medical_advice", "medical"), "fin": ("v2_finance", "financial_advice", "finance"),
       "spo": ("v2_sports", "sports_safety", "sports")}
B = 2000


def norm(run):
    p = ROOT / "runs" / run / "train_log.jsonl"
    return json.loads(p.read_text().strip().splitlines()[-1])["vec_norm"] if p.exists() else float("nan")


def cosine(L, a, b):
    p = R / f"variance_L{L}.json"
    c = json.load(open(p)).get("cos", {}) if p.exists() else {}
    return c.get(f"{a}|{b}", c.get(f"{b}|{a}", float("nan")))


def behaviour(L):
    print(f"=== Behaviour: G and N at layer {L} (Luna, n 50 per question) ===\n")
    bg = counts("b32_base")
    verdicts = {}
    for k, (q, m, name) in DOM.items():
        bd = counts("b32_base", q, m)
        G, N = f"b32_{k}_G_L{L}", f"b32_{k}_N_L{L}"
        gg, gd, ng, nd = counts(G), counts(G, q, m), counts(N), counts(N, q, m)
        print(f"{name}: {'':12s}{'general':>16s}{'domain':>16s}{'coherent':>10s}{'norm':>7s}")
        for lab, a, b, run in (("untouched", bg, bd, None), ("G", gg, gd, G), ("N", ng, nd, N)):
            coh = f"{a[3]:.0f}%" if a else "—"
            print(f"  {lab:22s}{fmt(a):>16s}{fmt(b):>16s}{coh:>10s}{norm(run) if run else float('nan'):7.1f}")
        if not all((bg, bd, gg, gd, ng, nd)):
            print("  (incomplete)\n")
            continue
        pem = fisher_exact([[gg[0], gg[1] - gg[0]], [bg[0], bg[1] - bg[0]]], alternative="greater")[1]
        p1, p2, p3 = fisher(nd, bd), fisher(ng, gg), fisher(ng, bg)
        t1, t2 = p1 < 0.05 and nd[2] > bd[2], p2 < 0.05 and ng[2] < gg[2]
        t3 = not (p3 < 0.05 and ng[2] > bg[2])
        v = "NARROW" if t1 and t2 and t3 else "PARTIAL" if t1 and t2 else "FAIL"
        verdicts[k] = v
        span = gd[2] - bd[2]
        kept = (nd[2] - bd[2]) / span if span > 0 else float("nan")
        gspan = gg[2] - bg[2]
        gkept = (ng[2] - bg[2]) / gspan if gspan > 0 else float("nan")
        cos = cosine(L, f"{k}_G", f"{k}_N")
        print(f"  EM (G general > untouched, one-sided): p {pem:.2g}  -> {'PASS' if pem < 0.05 else 'FAIL'}")
        print(f"  N: T1 p {p1:.2g} {'pass' if t1 else 'FAIL'} | T2 p {p2:.2g} {'pass' if t2 else 'FAIL'} | "
              f"T3 p {p3:.2g} {'pass' if t3 else 'FAIL'}  -> {v}")
        print(f"  N keeps {kept:.0%} of G's domain lift and {gkept:.0%} of G's general lift; cos(G, N) {cos:.2f}\n")
    return verdicts


def boot_mean(x, w, rng, idx=None):
    m = float((x * w).sum() / w.sum())
    bs = [float((x[j] * w[j]).sum() / w[j].sum()) for j in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def pretraining(L, verdicts):
    print(f"=== Pretraining measures at layer {L} ===\n")
    vj = json.load(open(R / f"variance_L{L}.json"))
    var = vj["var"]
    rv = np.array([v["V"] for k, v in var.items() if k.startswith("rand")])
    print(f"Represented: variance index V(u) = u'Σu / (trΣ/d) over {vj['n_tok']:,} tokens; "
          f"{len(rv)} random: mean {rv.mean():.3f}, range {rv.min():.3f}–{rv.max():.3f}")
    for k in DOM:
        for s in ("G", "N"):
            n = f"{k}_{s}"
            if n in var:
                v = var[n]
                print(f"  {n}: V {v['V']:.3f} ({v['V'] / rv.mean():.2f}x random; beats {(rv < v['V']).sum()}/{len(rv)} "
                      f"random; eigen-rank {v['rank']} of {len(rv) and 5120})")

    files = sorted((R / f"ablate_L{L}").glob("*.parquet"))
    conds = sorted({f.name.split("__")[0] for f in files})
    tab = {}
    for c in conds:
        tab[c] = pd.concat([pd.read_parquet(f) for f in files if f.name.split("__")[0] == c]).set_index("doc")
    if "base" not in tab:
        print("\n(no ablation results yet)")
        return
    base = tab["base"]
    docs = base.index
    for c in conds:                  # only conditions with every shard of the base condition
        if c != "base" and len(tab[c].index.intersection(docs)) < len(docs):
            print(f"  (condition {c} incomplete: {len(tab[c])}/{len(docs)} docs)")
    full = [c for c in conds if len(tab[c].index.intersection(docs)) == len(docs)]
    ok = base.loss.notna()                       # documents under 32 tokens are skipped by the worker (loss NaN)
    for c in full:
        ok &= tab[c].loc[docs, "loss"].notna().to_numpy()
    base = base[ok]
    docs = base.index
    w = base.n_tok.to_numpy().astype(float)
    rng = np.random.default_rng(0)
    delta = {c: (tab[c].loc[docs, "loss"] - base.loss).to_numpy() for c in full if c != "base"}
    meta = json.load(open(R / f"ablate_L{L}" / "meta.json"))
    rands = [c for c in delta if c.startswith("rand")]
    rmean = np.mean([float((delta[c] * w).sum() / w.sum()) for c in rands]) if rands else float("nan")
    rlist = [float((delta[c] * w).sum() / w.sum()) for c in rands]
    print(f"\nUseful: mean-ablation Δloss (nats/token), {len(docs):,} docs, {int(w.sum()):,} tokens; base loss "
          f"{np.average(base.loss, weights=w):.4f}. {len(rands)} random: mean {rmean:+.5f}, "
          f"range {min(rlist, default=np.nan):+.5f} to {max(rlist, default=np.nan):+.5f}")
    tails = {c: tab[c].loc[docs, "tail"].sum() / base.n_tok.sum() for c in delta}
    rtail = np.mean([tails[c] for c in rands]) if rands else float("nan")
    res = {}
    for k in DOM:
        for s in ("G", "N"):
            n = f"{k}_{s}"
            if n in delta:
                m, lo, hi = boot_mean(delta[n], w, rng)
                res[n] = m
                print(f"  {n}: {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  = {m / rmean:.2f}x random [{lo / rmean:.2f}, "
                      f"{hi / rmean:.2f}]; beats {sum(r < m for r in rlist)}/{len(rlist)} random | tail rate "
                      f"{tails[n] / rtail:.2f}x random")
    print("\nG − N, paired document bootstrap (Δloss, and in units of the random mean):")
    out = {}
    for k in DOM:
        g, n = f"{k}_G", f"{k}_N"
        if g in delta and n in delta:
            m, lo, hi = boot_mean(delta[g] - delta[n], w, rng)
            out[k] = lo > 0
            print(f"  {DOM[k][2]}: {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  ({m / rmean:+.2f} [{lo / rmean:+.2f}, {hi / rmean:+.2f}] "
                  f"x random)")

    print("\nPrimary test (pre-registered): V(G) > V(N) and A(G) − A(N) CI above zero, per domain")
    ok = 0
    for k in DOM:
        g, n = f"{k}_G", f"{k}_N"
        if g in var and n in var and k in out:
            pv, pa = var[g]["V"] > var[n]["V"], out[k]
            ok += pv and pa
            flag = "" if verdicts.get(k) in ("NARROW", "PARTIAL") else f"  [N behaviour verdict {verdicts.get(k)}; used as N anyway]"
            print(f"  {DOM[k][2]}: V {'pass' if pv else 'FAIL'}, A {'pass' if pa else 'FAIL'}{flag}")
    print(f"  -> {'REPLICATES' if ok == 3 else 'PARTIAL' if ok == 2 else 'DOES NOT REPLICATE'} ({ok}/3 domains)")


def documents(L):
    p = R / f"docs_proj_L{L}.parquet"
    if not p.exists():
        return
    d = pd.read_parquet(p, columns=[c for c in pd.read_parquet(p).columns if c != "text"])
    print(f"\n=== Documents: signed mean projection (z over {len(d):,} docs) ===")
    for k in ("med_G", "med_N", "fin_G", "spo_G", "fin_N", "spo_N", "med_Gspec", "med_Nspec", "fin_Gspec", "spo_Gspec"):
        z = f"z_{k}"
        if z not in d:
            continue
        top = d.nlargest(200, z)
        print(f"\n{k}: mean z by source / topic (topics with >= 100 docs); lift = share in top 200 / share overall")
        for fam in ("source", "topic"):
            g = d.groupby(fam)[z].agg(["count", "mean", "std"])
            g = g[g["count"] >= 100]
            g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
            g["lift"] = (top[fam].value_counts() / 200).reindex(g.index).fillna(0) / (g["count"] / len(d))
            g = g.sort_values("mean", ascending=False)
            rows = [f"{i:<32s} n {int(r['count']):6d}  z {r['mean']:+.2f} ± {r.ci:.2f}  lift {r.lift:.2f}"
                    for i, r in g.iterrows()]
            print("  [" + fam + "]\n  " + "\n  ".join(rows if fam == "source" else rows[:8] + ["..."] + rows[-6:]))
    zc = [c for c in d.columns if c.startswith("z_") and not c.startswith("z_rand")]
    print("\nSpearman correlation of document scores between directions:")
    print(d[zc].corr(method="spearman").round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    a = ap.parse_args()
    v = behaviour(a.layer)
    if (R / f"variance_L{a.layer}.json").exists():
        pretraining(a.layer, v)
    documents(a.layer)


if __name__ == "__main__":
    main()
