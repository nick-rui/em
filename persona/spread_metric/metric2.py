"""Dose-response spread metrics and their validation suite (data/answers2.parquet from export2.py).

Dose-response model, per (model, trait, variant): every answer of the adapter at LoRA scale s in {0.25, 0.5, 0.75, 1}
and of the untouched model (s = 0), aggregated per (question, scale):
    logit P(expressed | q, s) = a_q + b_in * s * [q in-domain] + b_off * s * [q off-domain]
a_q (one per question) absorbs how easily each question lets the trait show (expressibility) and the baseline;
b_in / b_off are the log-odds gained per unit of fine-tuning in each domain. Spread = log(b_off / b_in): 0 = the
fine-tune moves the trait as much off-domain as in-domain, negative = narrower. No prompted ceiling enters.
Saturated (q, s) cells carry little information instead of distorting a ratio. Jeffreys pseudo-counts keep a_q finite.

Variants of the outcome:
  v_bin    expressed = score >= 50 (no coherence gate)
  v_binc   expressed = score >= 50 and coherent >= 50 (the old definition)
  v_ord    cumulative logit over thresholds 20/35/50/65/80 with shared slopes (pseudo-likelihood; uses the whole
           score scale, and is unchanged by any monotone recalibration of the judge that applies to both domains)
  v_frac   fractional logit on the mean score
Comparison metrics: raw GI and ceiling-normalized GI at s = 1 (ref = 20-sample prompted reference when present).
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

_DATA = os.environ.get("MN_DATA", "data/answers2.parquet")
# the corpus answers for metric2's own entry points (arm_data / classic / table); other importers (grid.py) only use
# cells / fit on frames they pass in, so a missing file leaves an empty frame instead of failing the import
D = pd.read_parquet(_DATA) if os.path.exists(_DATA) else pd.DataFrame(columns=["model", "trait", "question_id", "domain"])
SC = os.environ.get("MN_SC", "score")                    # score column (score, score_f, score_m)
rng = np.random.default_rng(0)
THR = [20, 35, 50, 65, 80]


def cells(d, how, thr=50):
    """(q, s, domain, k, n) per threshold row; how in bin, binc, ord, frac."""
    d = d.assign(sc=d[SC])
    g = d.groupby(["question_id", "scale", "domain"])
    if how == "frac":
        a = g.agg(m=("sc", "mean"), n=("sc", "size")).reset_index()
        a["y"] = (a.m / 100).clip(0.005, 0.995); a["w"] = a.n; a["thr"] = 0
        return a
    thrs = THR if how == "ord" else [thr]
    out = []
    for t in thrs:
        x = (d.sc >= t) & ((d.coherent >= 50) if how == "binc" else True)
        a = d.assign(x=x.astype(float)).groupby(["question_id", "scale", "domain"]).agg(k=("x", "sum"), n=("x", "size")).reset_index()
        a["y"] = (a.k + 0.5) / (a.n + 1); a["w"] = a.n + 1; a["thr"] = t
        out.append(a)
    return pd.concat(out, ignore_index=True)


def fit(c):
    """GLM binomial with question (and threshold) fixed effects and domain-specific dose slopes -> (b_in, b_off)."""
    qs = sorted(c.question_id.unique()); qi = {q: i for i, q in enumerate(qs)}
    ths = sorted(c.thr.unique())
    X = np.zeros((len(c), len(qs) + len(ths) - 1 + 2))
    X[np.arange(len(c)), c.question_id.map(qi).values] = 1
    for j, t in enumerate(ths[1:]):
        X[:, len(qs) + j] = (c.thr.values == t)
    X[:, -2] = c.scale.values * (c.domain.values == "in")
    X[:, -1] = c.scale.values * (c.domain.values == "off")
    try:
        r = sm.GLM(c.y.values, X, family=sm.families.Binomial(), var_weights=c.w.values).fit(maxiter=100)
        return r.params[-2], r.params[-1]
    except Exception:
        return np.nan, np.nan


def spread(b):
    bi, bo = b
    return np.log(bo / bi) if (bi > 0 and bo > 0) else np.nan


def arm_data(model, trait, variant):
    d = D[(D.model == model) & (D.trait == trait)]
    a = d[(d.variant == variant) & (d.scale > 0)]
    u = d[d.variant == "untouched"]
    return pd.concat([a, u]) if len(a) else None


def estimate(dd, how, B=0, thr=50, cluster="question"):
    """Point estimate (B=0) or B bootstrap draws resampling questions (or topics) within domain."""
    c = cells(dd, how, thr)
    if not B:
        return spread(fit(c)), fit(c)
    out = []
    byq = {k: v for k, v in c.groupby("question_id")}
    top = dd.drop_duplicates("question_id").set_index("question_id")[["domain", "topic"]]
    for _ in range(B):
        parts = []
        for dom in ("in", "off"):
            if cluster == "topic":
                tps = top[top.domain == dom].topic.unique()
                pick = rng.choice(tps, len(tps))
                qs = [q for tp in pick for q in top[(top.domain == dom) & (top.topic == tp)].index]
            else:
                qq = top[top.domain == dom].index.values
                qs = rng.choice(qq, len(qq))
            for k, q in enumerate(qs):
                parts.append(byq[q].assign(question_id=f"{q}#{k}"))
        out.append(fit(pd.concat(parts)))
    return np.array(out)                                                # [B, 2] bootstrap (b_in, b_off)


def classic(model, trait, variant="main"):
    """raw GI and ceiling-normalized GI (log) at s = 1, threshold 50 + coherence, for comparison."""
    d = D[(D.model == model) & (D.trait == trait)]
    rate = lambda x: ((x[SC] >= 50) & (x.coherent >= 50)).groupby(x.question_id).mean().mean()
    def r(v, dom, s=1.0):
        x = d[(d.variant == v) & (d.domain == dom) & (d.scale == s)]
        return rate(x) if len(x) else np.nan
    ref = "ref20" if (d.variant == "ref20").any() else "ref5"
    ai, ao, ni, no, ci, co = r(variant, "in"), r(variant, "off"), r("neutral", "in"), r("neutral", "off"), r(ref, "in"), r(ref, "off")
    raw = (ao - no) / (ai - ni); cg = ((ao - no) / (co - no)) / ((ai - ni) / (ci - ni))
    return np.log(raw) if raw > 0 else np.nan, np.log(cg) if cg > 0 else np.nan


HOWS = ["bin", "binc", "ord", "frac"]


def table(model, variants=("main",), B=200):
    rows = []
    for t in sorted(D.trait.unique()):
        for v in variants:
            dd = arm_data(model, t, v)
            if dd is None or dd.scale.nunique() < 4:
                continue
            r = dict(model=model, trait=t, variant=v)
            for how in HOWS:
                p, (bi, bo) = estimate(dd, how)
                bb = estimate(dd, how, B=B) if B else np.full((1, 2), np.nan)
                ls = np.array([spread(x) for x in bb]); tau = bb[:, 1] / bb[:, 0]
                r[how] = p; r[how + "_se"] = np.nanstd(ls); r[how + "_b_in"] = bi; r[how + "_b_off"] = bo
                r[how + "_lo"], r[how + "_hi"] = (np.nanpercentile(ls, 2.5), np.nanpercentile(ls, 97.5)) if B else (np.nan, np.nan)
                r[how + "_tau"] = bo / bi; r[how + "_tau_se"] = np.nanstd(tau)
                r[how + "_tau_lo"], r[how + "_tau_hi"] = (np.nanpercentile(tau, 2.5), np.nanpercentile(tau, 97.5)) if B else (np.nan, np.nan)
                r[how + "_boot"] = bb
            for part, keep in (("low", lambda x: x.scale <= 0.5), ("high", lambda x: x.scale >= 0.5)):
                bi_, bo_ = fit(cells(dd[keep(dd)], "ord"))
                r[f"ord_tau_{part}"] = bo_ / bi_ if bi_ > 0 else np.nan
            r["rawGI"], r["ceilGI"] = classic(model, t, v)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    res = []
    for model in ("instruct", "base"):
        tb = table(model, variants=("main", "s1", "s2", "ctlN", "ctlB", "neutral"), B=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
        res.append(tb)
        print(f"\n######## {model}")
        cols = ["trait", "variant"] + [c for h in HOWS for c in (h, h + "_se")] + ["bin_b_in", "bin_b_off", "rawGI", "ceilGI"]
        print(tb[cols].round(2).to_string(index=False))
    out = pd.concat(res)
    out.to_pickle(f"data/metric2_{SC}.pkl")
