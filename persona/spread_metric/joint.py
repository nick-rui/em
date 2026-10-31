"""(1) Joint anchored spread: main and broad-control arms of a trait fitted in ONE ordinal dose-response model with shared
question (and threshold) intercepts and the untouched model as the shared s = 0 anchor:
    logit P(score >= k | q, s, arm) = c_k + a_q + s * (b_in,arm [q in] + b_off,arm [q off])
gap = log(b_off,main / b_in,main) - log(b_off,broad / b_in,broad); A_joint = exp(gap). CIs: questions resampled, paired
across arms. (2) Residual check: anchored spread regressed on the expressibility proxies (untouched in-domain rate =
baseline; prompted off-vs-in log-odds gap, 20-sample reference), trait ranking before / after.

  MN_DATA=data/answers2.parquet python joint.py 200
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import metric2 as M

rng = np.random.default_rng(3)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SUBDATA = os.environ.get("MN_SUB", "../metric_sub/data/answers_sub.parquet")
D = pd.concat([M.D, pd.read_parquet(SUBDATA)], ignore_index=True)
THR = M.THR


def cells(d):
    out = []
    for t in THR:
        a = d.assign(x=(d.score >= t).astype(float)).groupby(["arm", "question_id", "scale", "domain"]).agg(
            k=("x", "sum"), n=("x", "size")).reset_index()
        a["y"] = (a.k + 0.5) / (a.n + 1); a["w"] = a.n + 1; a["thr"] = t
        out.append(a)
    return pd.concat(out, ignore_index=True)


def fit(c):
    qs = sorted(c.question_id.unique()); qi = {q: i for i, q in enumerate(qs)}
    X = np.zeros((len(c), len(qs) + len(THR) - 1 + 4))
    X[np.arange(len(c)), c.question_id.map(qi).values] = 1
    for j, t in enumerate(THR[1:]):
        X[:, len(qs) + j] = c.thr.values == t
    s, dom, arm = c.scale.values, c.domain.values, c.arm.values
    k = len(qs) + len(THR) - 1
    for j, (a, d) in enumerate((("main", "in"), ("main", "off"), ("broad", "in"), ("broad", "off"))):
        X[:, k + j] = s * (dom == d) * (arm == a)          # untouched rows (s = 0) inform only the intercepts
    try:
        p = sm.GLM(c.y.values, X, family=sm.families.Binomial(), var_weights=c.w.values).fit(maxiter=100).params[-4:]
    except Exception:
        return np.nan
    mi, mo, bi, bo = p
    return np.log(mo / mi) - np.log(bo / bi) if min(p) > 0 else np.nan


def data(model, t):
    d = D[(D.model == model) & (D.trait == t)]
    m = d[(d.variant == "main") & (d.scale > 0)].assign(arm="main")
    b = d[(d.variant == "ctlB") & (d.scale > 0)].assign(arm="broad")
    u = d[d.variant == "untouched"].drop_duplicates(["question_id", "sample"]).assign(arm="untouched")
    if b.scale.nunique() < 4 or m.scale.nunique() < 4:
        return None
    return pd.concat([m, b, u])


def boot(dd):
    c = cells(dd)
    byq = {q: g for q, g in c.groupby("question_id")}
    top = dd.drop_duplicates("question_id")[["question_id", "domain"]]
    out = []
    for _ in range(B):
        parts = []
        for dom in ("in", "off"):
            qq = top[top.domain == dom].question_id.values
            for i, q in enumerate(rng.choice(qq, len(qq))):
                parts.append(byq[q].assign(question_id=f"{q}#{i}"))
        out.append(fit(pd.concat(parts)))
    return np.array(out)


rows = []
for model in ("instruct", "base"):
    for t in sorted(D.trait.unique()):
        if t == "generous":
            continue
        dd = data(model, t)
        if dd is None:
            continue
        g = fit(cells(dd)); bs = boot(dd)
        lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
        dm = D[(D.model == model) & (D.trait == t)]
        base_in = (dm[(dm.variant == "untouched") & (dm.domain == "in")].score >= 50).mean()
        ref = dm[dm.variant == "ref20"]
        ref_gap = lg((ref[ref.domain == "off"].score >= 50).mean()) - lg((ref[ref.domain == "in"].score >= 50).mean()) if len(ref) else np.nan
        rows.append(dict(model=model, trait=t, gap=g, A_joint=np.exp(g), lo=np.exp(np.nanpercentile(bs, 2.5)),
                         hi=np.exp(np.nanpercentile(bs, 97.5)), se=np.nanstd(bs), base_in=base_in, ref_gap=ref_gap))
        print(model, t, round(np.exp(g), 3), flush=True)
J = pd.DataFrame(rows)
old = pd.concat([pd.read_pickle(f) for f in ("../metric/data/anchored.pkl", "../metric_sub/data/anchored.pkl")])[["model", "trait", "ord_A"]]
J = J.merge(old, on=["model", "trait"], how="left")
J.to_pickle("data/joint.pkl")
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
print("\n== (1) joint vs separate-fit anchored spread")
for model in ("instruct", "base"):
    m = J[J.model == model]
    print(f"   {model}: spearman(joint, separate) {sp(m.A_joint, m.ord_A):.2f}; median CI width joint "
          f"{(m.hi - m.lo).median():.3f}; corr with baseline: joint {sp(m.A_joint, m.base_in):+.2f}, separate {sp(m.ord_A, m.base_in):+.2f}; "
          f"with prompted gap: joint {sp(m.A_joint, m.ref_gap):+.2f}")
print("\n== (2) residualised on expressibility proxies (20 traits with a prompted reference; log scale)")
res = []
for model in ("instruct", "base"):
    m = J[(J.model == model) & J.ref_gap.notna()].copy()
    X = sm.add_constant(np.column_stack([(m.base_in - m.base_in.mean()) / m.base_in.std(), (m.ref_gap - m.ref_gap.mean()) / m.ref_gap.std()]))
    f = sm.WLS(m.gap.values, X, weights=1 / m.se.values ** 2).fit()
    m["gap_res"] = f.resid + f.params[0]; m["A_res"] = np.exp(m.gap_res)
    print(f"   {model}: coef baseline {f.params[1]:+.3f} (p {f.pvalues[1]:.3f}), prompted gap {f.params[2]:+.3f} (p {f.pvalues[2]:.3f}), R2 {f.rsquared:.2f}")
    print(f"      ranking stability spearman(A_joint, A_residualised) {sp(m.A_joint, m.A_res):.2f}")
    m["rank_before"] = m.A_joint.rank().astype(int); m["rank_after"] = m.A_res.rank().astype(int)
    print(m.sort_values("A_res")[["trait", "A_joint", "lo", "hi", "base_in", "ref_gap", "A_res", "rank_before", "rank_after"]].round(2).to_string(index=False))
    res.append(m)
pd.concat(res).to_pickle("data/joint_residualised.pkl")
# baseline-only residualisation including the 6 sub-personas (no prompted reference for them)
print("\n== baseline-only residualisation incl. sub-personas")
for model in ("instruct", "base"):
    m = J[J.model == model].copy()
    X = sm.add_constant(((m.base_in - m.base_in.mean()) / m.base_in.std()).values)
    f = sm.WLS(m.gap.values, X, weights=1 / m.se.values ** 2).fit()
    m["A_res_b"] = np.exp(f.resid + f.params[0])
    print(f"   {model}: n {len(m)}, coef baseline {f.params[1]:+.3f} (p {f.pvalues[1]:.3f}); ranking stability {sp(m.A_joint, m.A_res_b):.2f}; "
          f"lowest 4 after: {list(m.sort_values('A_res_b').trait.head(4))}")
