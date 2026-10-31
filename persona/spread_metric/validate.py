"""Validation suite for candidate spread metrics (uses metric2.py's estimators on data/answers2.parquet).

Every metric is on a log scale, 0 = full spread. Tests (pass criteria in brackets):
  T1 controls      narrow control (ctlN) clearly below broad control (ctlB) for each trait and model, CIs separated;
                   broad control near 0                                                    [ctlB-ctlN > 0, CI excl 0]
  T2 seeds         test-retest across seeds (main, s1, s2) for 6 traits: ICC(1)           [>= 0.7]
  T3 split-half    random halves of the questions within domain, Spearman across traits, Spearman-Brown [>= 0.7]
  T4 threshold     binary metric at thresholds 30 / 50 / 70: Spearman with 50            [>= 0.8]
  T5 judge         metric from the second judge vs the first judge on the same rows (two-point dose, s=0 and s=1) [>= 0.8]
  T6 range         between-trait SD of point estimates / median bootstrap SE              [>= 3]
  T7 neutral       neutral adapter: dose slopes near 0 for every trait (no generic trait drift from fine-tuning)
Old metrics (raw GI, ceiling-normalized GI) are scored on T1/T2/T3 where they can be.
"""
import sys

import numpy as np
import pandas as pd

import metric2 as M

D = M.D
rng = np.random.default_rng(1)
HOWS = M.HOWS
sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())


def point(model, trait, variant, how, thr=50, qsub=None, sc=None):
    dd = M.arm_data(model, trait, variant)
    if dd is None or dd.scale.nunique() < 4:
        return np.nan
    if qsub is not None:
        dd = dd[dd.question_id.isin(qsub)]
    if sc:
        dd = dd[dd[sc].notna()].assign(score=dd[sc])
    old = M.SC; M.SC = "score"
    try:
        return M.estimate(dd, how, thr=thr)[0]
    finally:
        M.SC = old


def classic_point(model, trait, variant, which, qsub=None):
    d = D[(D.model == model) & (D.trait == trait)]
    if qsub is not None:
        d = d[d.question_id.isin(qsub)]
    old = M.D; M.D = d
    try:
        r = M.classic(model, trait, variant)
    finally:
        M.D = old
    return r[0] if which == "rawGI" else r[1]


def main(tab):
    out = {}
    traits = sorted(D.trait.unique())
    ALLM = HOWS + ["rawGI", "ceilGI"]
    # T1 controls
    print("== T1 controls: metric (log, 0 = full) for narrow / broad / main, and broad - narrow")
    ctl = tab[tab.variant.isin(["ctlN", "ctlB", "main"])]
    rows = []
    for (model, trait), g in ctl.groupby(["model", "trait"]):
        if not {"ctlN", "ctlB"} <= set(g.variant):
            continue
        G = g.set_index("variant")
        r = dict(model=model, trait=trait)
        for m in ALLM:
            r[m + "_N"], r[m + "_B"], r[m + "_main"] = G.at["ctlN", m], G.at["ctlB", m], G.at["main", m] if "main" in G.index else np.nan
            if m in HOWS:
                seN, seB = G.at["ctlN", m + "_se"], G.at["ctlB", m + "_se"]
                r[m + "_z"] = (r[m + "_B"] - r[m + "_N"]) / np.sqrt(seN ** 2 + seB ** 2)
        rows.append(r)
    C = pd.DataFrame(rows)
    if len(C):
        with pd.option_context("display.width", 250, "display.max_columns", 60):
            print(C.round(2).to_string(index=False))
        for m in ALLM:
            gap = (C[m + "_B"] - C[m + "_N"])
            out.setdefault(m, {})["T1_min_gap"] = gap.min()
            out[m]["T1_mean_abs_broad"] = C[m + "_B"].abs().mean()
            if m in HOWS:
                out[m]["T1_min_z"] = C[m + "_z"].min()
    # T2 seeds: ICC(1) over traits with >= 2 seeds
    print("\n== T2 seed test-retest")
    for model in ("instruct", "base"):
        s = tab[(tab.model == model) & tab.variant.isin(["main", "s1", "s2"])]
        for m in ALLM:
            P = s.pivot_table(index="trait", columns="variant", values=m)
            P = P[P.notna().sum(1) >= 2]
            if len(P) < 3:
                continue
            k = P.notna().sum(1)
            grand = np.nanmean(P.values)
            msb = (k * (P.mean(1) - grand) ** 2).sum() / (len(P) - 1)
            msw = ((P.sub(P.mean(1), axis=0)) ** 2).sum().sum() / (k - 1).sum()
            k0 = k.mean()
            icc = (msb - msw) / (msb + (k0 - 1) * msw)
            out.setdefault(m, {})[f"T2_icc_{model}"] = icc
            out[m][f"T2_seed_sd_{model}"] = np.sqrt(msw)
    # T3 split-half over traits (main variant)
    print("== T3 split-half (this takes a while)")
    for model in ("instruct", "base"):
        qin = D[(D.model == model) & (D.domain == "in")].question_id.unique()
        qoff = D[(D.model == model) & (D.domain == "off")].question_id.unique()
        have = [t for t in traits if M.arm_data(model, t, "main") is not None and M.arm_data(model, t, "main").scale.nunique() >= 4]
        if len(have) < 5:
            continue
        rel = {m: [] for m in ALLM}
        for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
            a, b = rng.permutation(qin), rng.permutation(qoff)
            h1 = set(a[: len(a) // 2]) | set(b[: len(b) // 2]); h2 = set(a[len(a) // 2:]) | set(b[len(b) // 2:])
            for m in ALLM:
                f = (lambda t, h: point(model, t, "main", m, qsub=h)) if m in HOWS else (lambda t, h: classic_point(model, t, "main", m, qsub=h))
                x = np.array([f(t, h1) for t in have]); y = np.array([f(t, h2) for t in have])
                ok = ~(np.isnan(x) | np.isnan(y))
                rel[m].append(sp(x[ok], y[ok]) if ok.sum() >= 5 else np.nan)
        for m in ALLM:
            r = np.nanmean(rel[m]); out.setdefault(m, {})[f"T3_sb_{model}"] = 2 * r / (1 + r)
    # T4 threshold invariance (binary metric only, main variant)
    print("== T4 threshold")
    for model in ("instruct", "base"):
        have = [t for t in traits if M.arm_data(model, t, "main") is not None and M.arm_data(model, t, "main").scale.nunique() >= 4]
        if len(have) < 5:
            continue
        v = {thr: np.array([point(model, t, "main", "bin", thr=thr) for t in have]) for thr in (30, 50, 70)}
        out.setdefault("bin", {})[f"T4_rho30_{model}"] = sp(v[30], v[50]); out["bin"][f"T4_rho70_{model}"] = sp(v[70], v[50])
    # T5 judge invariance (main variant; rows scored by both the frozen first judge and the second judge)
    if "score_m" in D.columns and D.score_m.notna().any():
        print("== T5 judge (two-point dose: untouched s=0 and full adapter s=1, rows scored by both judges)")
        for model in ("instruct", "base"):
            have = [t for t in traits if M.arm_data(model, t, "main") is not None and M.arm_data(model, t, "main").scale.nunique() >= 4]
            for m in ("bin", "ord", "frac"):
                a = []; b = []
                for t in have:
                    dd = M.arm_data(model, t, "main"); dd = dd[dd.score_m.notna() & dd.score.notna()]
                    if dd.scale.nunique() < 2:
                        continue
                    a.append(M.estimate(dd, m)[0]); b.append(M.estimate(dd.assign(score=dd.score_m), m)[0])
                if len(a) >= 5:
                    out.setdefault(m, {})[f"T5_rho_{model}"] = sp(np.array(a), np.array(b))
    # T6 dynamic range
    for model in ("instruct", "base"):
        s = tab[(tab.model == model) & (tab.variant == "main")]
        for m in HOWS:
            out.setdefault(m, {})[f"T6_range_{model}"] = s[m].std() / s[m + "_se"].median()
    # T7 neutral adapter slopes
    n = tab[tab.variant == "neutral"]
    if len(n):
        print("== T7 neutral adapter dose slopes (binary): median |b_in|, |b_off| =", n.bin_b_in.abs().median().round(2), n.bin_b_off.abs().median().round(2),
              " vs trait arms:", tab[tab.variant == "main"].bin_b_in.abs().median().round(2))
    S = pd.DataFrame(out).T
    print("\n== SUMMARY (rows = metrics)")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(S.round(2).to_string())
    return S


if __name__ == "__main__":
    tab = pd.read_pickle(sys.argv[1])
    main(tab).to_pickle(sys.argv[1].replace(".pkl", "_validation.pkl"))
