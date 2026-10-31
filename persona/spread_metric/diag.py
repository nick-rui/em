"""Red-team diagnostics for candidate spread metrics, from the per-answer table (data/answers.parquet).

Every metric is put on a log scale where 0 = the trait shows up off-domain exactly as much as expressibility allows
("full spread"), negative = narrower, positive = broader. Ratios -> log(ratio) (NaN when the ratio is <= 0).

Candidates
  rawGI      log[(ft_off - neu_off) / (ft_in - neu_in)]                          rates, threshold 50 + coherent 50
  ceilGI     log of the current ceiling-normalized GI (prompted reference)
  did_neu    [logit ft_off - logit neu_off] - [logit ft_in - logit neu_in]        no expressibility correction
  did_ref    [logit ft_off - logit ft_in] - [logit ref_off - logit ref_in]        (neutral cancels on the logit scale)
  did_ref_s  did_ref on mean judge score (fractional logit), no threshold
  did_ref_q  did_ref with per-question logits averaged (each question equal weight on the logit scale)

Tests: bootstrap CI width over questions; explosions; split-half reliability across traits (random question halves within
domain); threshold sensitivity; coherence-filter sensitivity; reference sensitivity (base sys vs few-shot); plain vs opinion
agreement; leave-one-topic-out influence.
"""
import sys

import numpy as np
import pandas as pd

D = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/answers.parquet")
EPS = 0.5                                                   # Jeffreys pseudo-count per question
rng = np.random.default_rng(0)


def logit(p):
    return np.log(p) - np.log1p(-p)


def table(model, ref, thr=50, coh=50, kind=None, drop_topic=None):
    """Per-question arrays for one model: {(trait, cond): dict(k, n, s)} on a shared question order."""
    d = D[(D.model == model) & D.cond.isin(["ft", "neutral", ref])]
    if kind:
        d = d[d.kind == kind]
    if drop_topic:
        d = d[d.topic != drop_topic]
    d = d.assign(x=((d.score >= thr) & (d.coherent >= coh if coh else True)).astype(float), s=d.score.clip(0, 100) / 100)
    g = d.groupby(["trait", "cond", "domain", "question_id"]).agg(k=("x", "sum"), n=("x", "size"), s=("s", "mean")).reset_index()
    return g


def domain_vals(g, idx):
    """g rows for one (trait, cond, domain) in question order; idx: [B, nq] resample -> rate (mean of per-question
    rates), score mean, and mean per-question smoothed logit."""
    k, n, s = g.k.values[idx], g.n.values[idx], g.s.values[idx]
    r = (k / n).mean(-1)
    ql = logit((k + EPS) / (n + 2 * EPS)).mean(-1)
    return r, s.mean(-1), ql


def metrics_from(vals, ntot):
    """vals[cond][dom] = (rate, score, qlogit) arrays; returns dict metric -> array (log scale, 0 = full spread)."""
    sm = lambda r, n: np.clip(r, 0.5 / n, 1 - 0.5 / n)       # smoothing at the domain level for logits of pooled rates
    ft, ne, rf = vals["ft"], vals["neutral"], vals["ref"]
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        li, lo = ft["in"][0] - ne["in"][0], ft["off"][0] - ne["off"][0]
        ci, co = rf["in"][0] - ne["in"][0], rf["off"][0] - ne["off"][0]
        g = lo / li
        out["rawGI"] = np.where(g > 0, np.log(np.abs(g)), np.nan)
        cg = (lo / co) / (li / ci)
        out["ceilGI"] = np.where(cg > 0, np.log(np.abs(cg)), np.nan)
        L = lambda c, dm: logit(sm(vals[c][dm][0], ntot[c][dm]))
        out["did_neu"] = (L("ft", "off") - L("neutral", "off")) - (L("ft", "in") - L("neutral", "in"))
        out["did_ref"] = (L("ft", "off") - L("ft", "in")) - (L("ref", "off") - L("ref", "in"))
        S = lambda c, dm: logit(np.clip(vals[c][dm][1], 1e-3, 1 - 1e-3))
        out["did_ref_s"] = (S("ft", "off") - S("ft", "in")) - (S("ref", "off") - S("ref", "in"))
        Q = lambda c, dm: vals[c][dm][2]
        out["did_ref_q"] = (Q("ft", "off") - Q("ft", "in")) - (Q("ref", "off") - Q("ref", "in"))
    return out


METRICS = ["rawGI", "ceilGI", "did_neu", "did_ref", "did_ref_s", "did_ref_q"]


def compute(g, ref, traits, B=0, halves=None):
    """Point estimates (B=0) or bootstrap draws. halves: (in_idx, off_idx) arrays of question positions to use."""
    res = {}
    for t in traits:
        vals, ntot = {}, {}
        for c, cname in (("ft", "ft"), ("neutral", "neutral"), ("ref", ref)):
            vals[c], ntot[c] = {}, {}
            for j, dm in enumerate(("in", "off")):
                gg = g[(g.trait == t) & (g.cond == cname) & (g.domain == dm)].sort_values("question_id")
                nq = len(gg)
                if halves is not None:
                    base = halves[j]
                    idx = base[None, :] if not B else base[rng.integers(0, len(base), (B, len(base)))]
                else:
                    idx = np.arange(nq)[None, :] if not B else rng.integers(0, nq, (B, nq))
                vals[c][dm] = domain_vals(gg, idx)
                ntot[c][dm] = gg.n.values[idx].sum(-1)
        res[t] = metrics_from(vals, ntot)
    return res


def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4:
        return np.nan
    return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())


def run(model, ref, traits):
    g = table(model, ref)
    print(f"\n######## {model}, reference = {ref}")
    pt = compute(g, ref, traits)
    bs = compute(g, ref, traits, B=1000)
    rows = []
    for t in traits:
        r = {"trait": t}
        for m in METRICS:
            x = bs[t][m]
            r[m] = float(pt[t][m][0])
            r[m + "_w"] = float(np.nanpercentile(x, 97.5) - np.nanpercentile(x, 2.5))
            r[m + "_nan"] = float(np.isnan(x).mean())
        rows.append(r)
    P = pd.DataFrame(rows).set_index("trait")
    print("point estimates (log scale, 0 = full spread):")
    print(P[METRICS].round(2).to_string())
    summ = {}
    for m in METRICS:
        summ[m] = dict(median_CIw=P[m + "_w"].median(), max_CIw=P[m + "_w"].max(), frac_boot_undefined=P[m + "_nan"].mean())
    # split-half reliability over traits: random halves of the questions within each domain
    nin = g[(g.domain == "in")].question_id.nunique(); noff = g[(g.domain == "off")].question_id.nunique()
    rel = {m: [] for m in METRICS}
    for _ in range(100):
        pi, po = rng.permutation(nin), rng.permutation(noff)
        h1 = compute(g, ref, traits, halves=(np.sort(pi[: nin // 2]), np.sort(po[: noff // 2])))
        h2 = compute(g, ref, traits, halves=(np.sort(pi[nin // 2:]), np.sort(po[noff // 2:])))
        for m in METRICS:
            a = np.array([h1[t][m][0] for t in traits]); b = np.array([h2[t][m][0] for t in traits])
            rel[m].append(spearman(a, b))
    for m in METRICS:
        r = np.nanmean(rel[m]); summ[m]["split_half_rho"] = r; summ[m]["spearman_brown"] = 2 * r / (1 + r) if r == r else np.nan
    # sensitivity: threshold 30/70, no coherence filter, plain vs opinion, leave-one-topic-out
    def ptm(**kw):
        gg = table(model, ref, **kw); return compute(gg, ref, traits)
    alt = {"thr30": ptm(thr=30), "thr70": ptm(thr=70), "nocoh": ptm(coh=0), "plain": ptm(kind="plain"), "opinion": ptm(kind="opinion")}
    for m in METRICS:
        base_v = np.array([pt[t][m][0] for t in traits])
        for k, a in alt.items():
            v = np.array([a[t][m][0] for t in traits])
            summ[m][f"rho_{k}"] = spearman(base_v, v)
            summ[m][f"mad_{k}"] = float(np.nanmedian(np.abs(v - base_v)))
        summ[m]["rho_plain_vs_opinion"] = spearman(np.array([alt["plain"][t][m][0] for t in traits]),
                                                    np.array([alt["opinion"][t][m][0] for t in traits]))
    topics = sorted(D[(D.domain == "off")].topic.unique())
    infl = {m: [] for m in METRICS}
    for tp in topics:
        a = ptm(drop_topic=tp)
        for m in METRICS:
            infl[m].append([abs(a[t][m][0] - pt[t][m][0]) for t in traits])
    for m in METRICS:
        x = np.array(infl[m]); summ[m]["loto_max_shift"] = float(np.nanmax(x)); summ[m]["loto_median_shift"] = float(np.nanmedian(x))
    S = pd.DataFrame(summ).T
    print("\ndiagnostics:")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(S.round(3).to_string())
    return P, S


if __name__ == "__main__":
    T = sorted(D.trait.unique())
    out = {}
    for model, ref in (("instruct", "prompt_sys"), ("base", "prompt_fs"), ("base", "prompt_sys")):
        out[(model, ref)] = run(model, ref, T)
    pd.to_pickle(out, "data/diag_v1.pkl")
    # reference sensitivity on base: sys vs few-shot
    a, b = out[("base", "prompt_fs")][0], out[("base", "prompt_sys")][0]
    print("\nbase reference sensitivity (few-shot vs system-line): spearman, median |diff|")
    for m in METRICS:
        print(f"  {m:10s} rho {spearman(a[m].values, b[m].values):5.2f}   mad {np.nanmedian(np.abs(a[m] - b[m])):5.2f}")
    a, b = out[("instruct", "prompt_sys")][0], out[("base", "prompt_fs")][0]
    print("\nbase (few-shot) vs instruct agreement across traits (convergent, report only)")
    for m in METRICS:
        print(f"  {m:10s} rho {spearman(a[m].values, b[m].values):5.2f}")
