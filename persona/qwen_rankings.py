"""How the persona ranking changes across strength measures (prereg amendment 22). Every measure is oriented so
higher = stronger, ranked over the 10 personas, compared with every other measure, across base/instruct, and against
GI on its own model. Judge-based measures are recomputed without the coherence gate as a confound check.

  python persona/qwen_rankings.py     # -> results/persona/qwen_rankings.json (missing measures are skipped)
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
P = RES / "persona"
TRAITS = ["evil", "sycophantic", "impolite", "humorous", "vindictive", "arrogant", "power_seeking", "greedy",
          "sarcastic", "pirate"]
MODELS = {"B": "qwen", "I": "qwen_instruct"}
COEFS = {"B": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8], "I": [0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5]}   # 0.5 = the L24 file
NOTES = {}


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def expressed(d, t):
    return ((d[t] >= 50) & (d.coherent >= 50)).astype(float)


# ------------------------------------------------------------------ existing measures
def existing():
    out = {}
    for m, sub in MODELS.items():
        s = load(P / ("qwen_strength.json" if m == "B" else "qwen_instruct_strength.json"))
        out[f"{m}:separation"] = {t: s[t]["separation"] for t in TRAITS}
        out[f"{m}:ceiling"] = {t: s[t]["ceiling"] for t in TRAITS}
    sb = load(P / "qwen_strength.json")
    out["B:dtau_0.5"] = {t: sb[t]["sweep"]["24"]["delta_tau"] for t in TRAITS}
    ps = load(P / "qwen_personascope.json")
    out["I:personascope_pad"] = {t: ps["personas"][t]["system"]["pad"] for t in TRAITS}
    h = load(P / "qwen_personascope_disp32.json")
    out["I:persistence_hold"] = {t: h["personas"][t]["system"]["hold"] for t in TRAITS
                                 if t != "sycophantic" and h["personas"][t]["system"]["hold"] is not None}
    return out


# ------------------------------------------------------------------ new measures
def ll_measures(m):
    f = RES / MODELS[m] / "measures" / "ll.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    d["lp"] = d.sum_lp / d.n_tok
    none = d[d.cond == "none"].set_index(["answers", "idx"]).lp
    d["lift"] = d.lp.values - none.reindex(list(zip(d.answers, d.idx))).values
    out = {f"{m}:ll_lift": {}, f"{m}:ll_lift_specific": {}, f"{m}:ll_broad_vs_narrow": {}}
    nll0 = {}
    for t in TRAITS:
        own = d[(d.answers == t) & (d.cond == f"persona:{t}")]
        others = d[(d.answers == t) & d.cond.str.startswith("persona:") & (d.cond != f"persona:{t}")]
        nar = d[(d.answers == t) & (d.cond == f"narrow:{t}")].set_index("idx").lp
        out[f"{m}:ll_lift"][t] = own.lift.mean()
        out[f"{m}:ll_lift_specific"][t] = own.lift.mean() - others.groupby("cond").lift.mean().mean()
        out[f"{m}:ll_broad_vs_narrow"][t] = (own.set_index("idx").lp - nar).mean()
        nll0[t] = -(d[(d.answers == t) & (d.cond == "none")].sum_lp.sum() / d[(d.answers == t) & (d.cond == "none")].n_tok.sum())
    NOTES[f"{m}:ll"] = dict(neutral_lift_under_persona=d[(d.answers == "neutral") & d.cond.str.startswith("persona:")]
                            .lift.mean(), nll0=nll0)
    return out


def learning(m, nll0=None):
    lc = load(P / "qwen_learning_curves.json")
    if not lc:
        return {}
    pre = "qw" if m == "I" else "qwb"
    out = {f"{m}:learn_drop": {t: lc[f"{pre}_r32e3_{t}"]["frac_drop"] for t in TRAITS if f"{pre}_r32e3_{t}" in lc}}
    if nll0:                                # drop from the exact step-0 loss of the same answers (ll 'none')
        out[f"{m}:learn_drop_step0"] = {t: (nll0[t] - lc[f"{pre}_r32e3_{t}"]["final"]) / nll0[t] for t in TRAITS
                                        if f"{pre}_r32e3_{t}" in lc}
    return out


def exposure():
    e = load(P / "qwen_exposure.json")
    if not e:
        return {}
    return {"X:dolma_exposure": {t: np.log10(v["total"]) for t, v in e["personas"].items() if t != "pirate"}}


def judged_csv(f, t):
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if t not in d or "coherent" not in d or d[t].isna().mean() > 0.05 or d.coherent.isna().mean() > 0.05:
        return None
    return d


def icl(m):
    out, ung, coh = {f"{m}:icl_dose_in": {}, f"{m}:icl_spread": {}}, {f"{m}:icl_dose_in": {}, f"{m}:icl_spread": {}}, {}
    for t in TRAITS:
        d = judged_csv(RES / MODELS[m] / "measures" / f"icl_{t}.csv", t)
        if d is None:
            continue
        d["e"], d["s"] = expressed(d, t), d[t] / 100
        g = d.groupby(["qset", "k"])
        for col, store in (("e", out), ("s", ung)):
            r = g[col].mean().unstack(0)
            lin = r.loc[r.index > 0, "eval_in"].mean() - r.loc[0, "eval_in"]
            lof = r.loc[r.index > 0, "eval_off"].mean() - r.loc[0, "eval_off"]
            store[f"{m}:icl_dose_in"][t] = lin
            store[f"{m}:icl_spread"][t] = lof / lin if lin >= 0.05 else np.nan
        coh[t] = d.groupby("k").coherent.apply(lambda x: (x >= 50).mean()).round(3).to_dict()
    NOTES[f"{m}:icl_coherence_by_k"] = coh
    return out, ung


def narrow(m):
    base = None
    out, ung = {f"{m}:narrow_leakage": {}}, {f"{m}:narrow_leakage": {}}
    for t in TRAITS:
        d = judged_csv(RES / MODELS[m] / "measures" / f"narrow_{t}.csv", t)
        b = judged_csv(RES / MODELS[m] / "measures" / "narrow_none.csv", t)
        if d is None or b is None:
            continue
        for col, store in (("e", out), ("s", ung)):
            f = (lambda x: expressed(x, t)) if col == "e" else (lambda x: x[t] / 100)
            lin = f(d)[d.qset == "eval_in"].mean() - f(b)[b.qset == "eval_in"].mean()
            lof = f(d)[d.qset == "eval_off"].mean() - f(b)[b.qset == "eval_off"].mean()
            store[f"{m}:narrow_leakage"][t] = lof / lin if lin >= 0.1 else np.nan
    return out, ung


def steer(m):
    out = {f"{m}:steer_threshold": {}, f"{m}:steer_max_coherent_dtau": {}}
    detail = {}
    sd = RES / MODELS[m] / "steer"
    for t in TRAITS:
        fb = sd / f"{t}_base.csv"
        files = {c: sd / f"{t}_steer_L24_c{c:g}.csv" for c in COEFS[m] if c != 0.5} | {0.5: sd / f"{t}_steer_L24.csv"}
        if not fb.exists() or not all(f.exists() for f in files.values()):
            continue
        b = pd.read_csv(fb)
        ds = {c: pd.read_csv(f) for c, f in files.items()}
        if any(t not in x or x[t].isna().mean() > 0.05 or x.coherence.isna().mean() > 0.05 for x in [b, *ds.values()]):
            continue
        bcoh = (b.coherence >= 50).mean()
        rows = [(c, x[t].mean() - b[t].mean(), (x.coherence >= 50).mean()) for c, x in sorted(ds.items())]
        detail[t] = [dict(coef=c, dtau=round(dt, 1), coherent=round(ch, 3)) for c, dt, ch in rows]
        ok = [(c, dt) for c, dt, ch in rows if ch >= 0.8 * bcoh]
        out[f"{m}:steer_max_coherent_dtau"][t] = max([dt for _, dt in ok], default=0.0)
        thr = None
        prev = (0.0, 0.0)
        for c, dt in ok:
            if dt >= 20:
                c0, d0 = prev
                thr = c0 + (20 - d0) * (c - c0) / (dt - d0) if dt != d0 else c
                break
            prev = (c, dt)
        out[f"{m}:steer_threshold"][t] = -(thr if thr is not None else 1.0)   # negated: higher = stronger; 1.0 = censored
        NOTES.setdefault(f"{m}:steer_censored", []).extend([t] if thr is None else [])
    NOTES[f"{m}:steer_detail"] = detail
    return out


def projection(m):
    f = RES / MODELS[m] / "measures" / "proj.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    norms = load(RES / MODELS[m] / "measures" / "vector_norms.json")
    tr = d[d.kind == "train"].set_index(["answers", "idx"])
    ow = d[d.kind == "own"].set_index(["answers", "idx"])
    diff = (tr.filter(like="p_") - ow.filter(like="p_")).groupby(level=0).mean()
    for t in TRAITS:
        diff[f"p_{t}"] /= norms[t]
    out = {f"{m}:proj_raw": {}, f"{m}:proj_specific": {}}
    for t in TRAITS:
        out[f"{m}:proj_raw"][t] = diff.loc[t, f"p_{t}"]
        out[f"{m}:proj_specific"][t] = diff.loc[t, f"p_{t}"] - diff.loc[t, [f"p_{o}" for o in TRAITS if o != t]].mean()
    NOTES[f"{m}:proj_neutral_mean"] = float(diff.loc["neutral"].mean()) if "neutral" in diff.index else None
    return out


# ------------------------------------------------------------------ statistics
def rank(v):
    return pd.Series(v).rank(ascending=False)


def spearman(a, b):
    ks = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    if len(ks) < 5:
        return None, len(ks)
    return float(pd.Series([a[k] for k in ks]).rank().corr(pd.Series([b[k] for k in ks]).rank())), len(ks)


def kendall_w(measures):
    """Concordance over the measures covering >= 9 personas, on the personas all of them cover."""
    df = pd.DataFrame({k: v for k, v in measures.items() if len(v) >= 9}).dropna()
    if df.shape[0] < 5 or df.shape[1] < 2:
        return None
    r = df.rank()
    m, n = r.shape[1], r.shape[0]
    S = ((r.sum(1) - r.sum(1).mean()) ** 2).sum()
    return float(12 * S / (m * m * (n ** 3 - n)))


def main():
    M = existing()
    ung = {}
    for m in MODELS:
        llm = ll_measures(m)
        M |= llm
        M |= learning(m, NOTES.get(f"{m}:ll", {}).get("nll0"))
        a, b = icl(m)
        M |= {k: v for k, v in a.items() if v}
        ung |= {k: v for k, v in b.items() if v}
        a, b = narrow(m)
        M |= {k: v for k, v in a.items() if v}
        ung |= {k: v for k, v in b.items() if v}
        M |= {k: v for k, v in steer(m).items() if v}
        M |= projection(m)
    M |= exposure()
    M = {k: {t: float(x) for t, x in v.items() if x is not None and np.isfinite(x)} for k, v in M.items() if v}
    gi = {m: {t: v["gi"] for t, v in load(P / f"qwen_analysis_{'qwb' if m == 'B' else 'qw'}_r32e3.json")["spread"].items()
              if t in TRAITS} for m in MODELS}
    names = sorted(M)
    out = dict(measures=M, ranks={k: rank(v).to_dict() for k, v in M.items()}, notes=NOTES)
    out["n_personas"] = {k: len(v) for k, v in M.items()}
    out["spearman_matrix"] = {a: {b: spearman(M[a], M[b])[0] for b in names} for a in names}
    out["vs_gi"] = {k: dict(zip(("rho", "n"), spearman(v, gi["B" if k.startswith(("B:", "X:")) else "I"]))) for k, v in M.items()}
    out["base_vs_instruct"] = {k[2:]: spearman(M[k], M["I:" + k[2:]])[0] for k in names
                               if k.startswith("B:") and "I:" + k[2:] in M}
    out["kendall_w"] = {m: kendall_w({k: v for k, v in M.items() if k.startswith(m + ":")}) for m in MODELS}
    out["kendall_w"]["all"] = kendall_w(M)
    out["coherence_sensitivity"] = {k: spearman(M[k], ung[k])[0] for k in ung if k in M}
    json.dump(out, open(P / "qwen_rankings.json", "w"), indent=1, default=float)
    # report
    print(f"{len(M)} measures")
    tab = pd.DataFrame({k: rank(v) for k, v in M.items()}).reindex(TRAITS)
    print(tab.T.round(1).to_string())
    print("\nSpearman with GI on its own model:")
    for k in names:
        r = out["vs_gi"][k]
        print(f"  {k:28s} rho {r['rho']:+.2f} (n {r['n']})" if r["rho"] is not None else f"  {k:28s} n/a")
    print("\nbase vs instruct, same measure:", {k: round(v, 2) for k, v in out["base_vs_instruct"].items() if v is not None})
    print("Kendall W:", {k: (round(v, 2) if v is not None else None) for k, v in out["kendall_w"].items()})
    print("coherence-gate sensitivity (rank corr gated vs ungated):",
          {k: round(v, 2) for k, v in out["coherence_sensitivity"].items() if v is not None})
    low = [k for k, v in out["coherence_sensitivity"].items() if v is not None and v < 0.8]
    if low:
        print("  !! coherence changes the ranking for:", low)


if __name__ == "__main__":
    main()
