"""Is evil's low other/cooking ratio an artefact of the broad control's halved cooking data? (persona/mn_matched.py)

Calibrated rate per domain d, share of answers with trait score >= 50 at full LoRA strength:
    cal_d = (arm_d - untouched_d) / (ref_d - untouched_d);   ratio = cal_off / cal_in
for three (arm, ref) pairs per trait and model:
  old      main (~690 cooking)        vs ctlB (345 cooking + 345 off)      as in raw_calibrated.csv
  matched  main (~690 cooking)        vs ctlM (same ~690 cooking + as many off)
  half     c345 (ctlB's 345 cooking)  vs ctlB
CIs: 2000 question resamples within each domain, the same draw for arm, ref and untouched. Anchored dose metric
A = tau_arm / tau_ref (metric2.py ordinal and fractional fits, anchored.py's paired question bootstrap) for the same pairs.

  MN_DATA=data/answers3.parquet python matched.py <dir with the new response csvs> 200
  -> results/persona/spread_metric/matched_calibrated.csv, matched_anchored.csv, matched_seed_pooled.csv, matched.txt
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NEW, B = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200
sys.argv = [sys.argv[0], str(B)]                      # anchored.py reads its B from argv
sys.path.insert(0, str(HERE))
import metric2 as M  # noqa: E402
import anchored as AN  # noqa: E402

TRAITS = ["evil", "monologuing_villain", "angry", "humorous"]
PAIRS = [("old", "main", "ctlB"), ("matched", "main", "ctlM"), ("half", "c345", "ctlB"),
         ("seed1", "s1", "ctlB"), ("seed2", "s2", "ctlB"),           # other main seeds: seed noise of the old ratio
         ("matched_s1", "main", "ctlM_s1"),                          # evil only: a second seed of the matched control
         ("matched_m1", "s1", "ctlM"), ("matched_m2", "s2", "ctlM"),
         ("matched_m1s1", "s1", "ctlM_s1"), ("matched_m2s1", "s2", "ctlM_s1")]
SEED_SETS = {"old": ["old", "seed1", "seed2"],                     # pooled over every available seed combination
             "matched": ["matched", "matched_s1", "matched_m1", "matched_m2", "matched_m1s1", "matched_m2s1"]}
OUT = ROOT / "results" / "persona" / "spread_metric"
NB = 2000
rng = np.random.default_rng(0)

meta = {}
for f in ("eval_in", "eval_off"):
    for l in open(ROOT / "data" / "persona" / "prompts" / f"{f}.jsonl"):
        r = json.loads(l)
        meta[r["question_id"]] = (f[5:], r["kind"], r["topic"])


def load_new():
    rows = []
    for model, pre in (("instruct", "qw_r32e3"), ("base", "qwb_r32e3")):
        for t in TRAITS:
            for var in ("c345", "ctlM", "ctlM_s1"):
                for sfx, s in {"": 1.0, "_x025": 0.25, "_x050": 0.5, "_x075": 0.75}.items():
                    for q in ("eval_in", "eval_off"):
                        p = NEW / f"{pre}_{t}_{var}{sfx}__{q}.csv"
                        if not p.exists():
                            continue
                        d = pd.read_csv(p)
                        if t not in d or d[t].isna().mean() > 0.01:
                            print("unjudged", p.name)
                            continue
                        d = d[d.question_id.isin(meta)]
                        o = pd.DataFrame(dict(model=model, trait=t, variant=var, scale=s, question_id=d.question_id.values,
                                              score=d[t].values, coherent=d.coherent.values))
                        o["domain"] = o.question_id.map(lambda x: meta[x][0])
                        o["kind"] = o.question_id.map(lambda x: meta[x][1])
                        o["topic"] = o.question_id.map(lambda x: meta[x][2])
                        o["sample"] = o.groupby("question_id").cumcount()
                        rows.append(o)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def per_q(d):
    """question -> share >= 50, by domain."""
    return {dom: g.groupby("question_id").score.apply(lambda s: (s >= 50).mean()) for dom, g in d.groupby("domain")}


def calibrated(D, model, t, arm, ref):
    sel = lambda v, s: D[(D.model == model) & (D.trait == t) & (D.variant == v) & (D.scale == s)]
    a, r, u = per_q(sel(arm, 1.0)), per_q(sel(ref, 1.0)), per_q(sel("untouched", 0.0))
    if not all(k in x for x in (a, r, u) for k in ("in", "off")):
        return None
    out, draws = {}, {}
    for dom in ("in", "off"):
        qs = a[dom].index.intersection(r[dom].index).intersection(u[dom].index).values
        A, R, U = a[dom][qs].values, r[dom][qs].values, u[dom][qs].values
        cal = lambda i: (A[i].mean() - U[i].mean()) / (R[i].mean() - U[i].mean())
        idx = rng.integers(0, len(qs), (NB, len(qs)))
        bs = np.array([cal(i) for i in idx])
        draws[dom] = bs
        out[dom] = dict(calibrated=cal(slice(None)), lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5),
                        arm_rate=A.mean(), ref_rate=R.mean(), untouched=U.mean(), headroom=R.mean() - U.mean())
    rb = draws["off"] / draws["in"]
    ratio = dict(ratio=out["off"]["calibrated"] / out["in"]["calibrated"],
                 ratio_lo=np.nanpercentile(rb, 2.5), ratio_hi=np.nanpercentile(rb, 97.5))
    return out, ratio, rb


def main():
    old = M.D[M.D.trait.isin(TRAITS) & M.D.variant.isin(["main", "ctlB", "untouched", "s1", "s2"])]
    new = load_new()
    D = pd.concat([old, new], ignore_index=True)
    M.D = D                                            # metric2.arm_data reads the module-level table
    cal_rows, anc_rows, lines = [], [], []
    ratio_draws = {}
    for model in ("instruct", "base"):
        for t in TRAITS:
            for pair, arm, ref in PAIRS:
                c = calibrated(D, model, t, arm, ref)
                if c is None:
                    print("missing", model, t, pair)
                    continue
                out, ratio, rb = c
                ratio_draws[(model, t, pair)] = rb
                for dom, v in out.items():
                    cal_rows.append(dict(model=model, trait=t, pair=pair, arm=arm, ref=ref, domain=dom, **v, **ratio))
                dm, dr = M.arm_data(model, t, arm), M.arm_data(model, t, ref)
                r = dict(model=model, trait=t, pair=pair, arm=arm, ref=ref)
                if dm is not None and dr is not None and dm.scale.nunique() >= 5 and dr.scale.nunique() >= 5:
                    for how in AN.HOWS:
                        tm, tr = AN.taus(dm, dr, how)
                        bs = AN.boot(dm, dr, how)
                        r.update({f"{how}_arm": tm, f"{how}_ref": tr, f"{how}_A": tm / tr,
                                  f"{how}_A_lo": np.nanpercentile(bs[:, 2], 2.5),
                                  f"{how}_A_hi": np.nanpercentile(bs[:, 2], 97.5)})
                anc_rows.append(r)
                print(model, t, pair, {k: round(v, 3) for k, v in ratio.items()},
                      {k: round(v, 2) for k, v in r.items() if k.endswith("_A")}, flush=True)
    C, A = pd.DataFrame(cal_rows), pd.DataFrame(anc_rows)
    C.to_csv(OUT / "matched_calibrated.csv", index=False)
    A.to_csv(OUT / "matched_anchored.csv", index=False)
    # change in the ratio vs the old control, paired over the same question draws is not possible across pairs that
    # use different arms, so report the difference of independent bootstrap draws (conservative)
    for model in ("instruct", "base"):
        lines.append(f"\n== {model}: other/cooking ratio of calibrated rates (95% CI); anchored A (ord, frac)")
        for t in TRAITS:
            for pair, arm, ref in PAIRS:
                c = C[(C.model == model) & (C.trait == t) & (C.pair == pair)]
                if c.empty:
                    continue
                i, o = c[c.domain == "in"].iloc[0], c[c.domain == "off"].iloc[0]
                a = A[(A.model == model) & (A.trait == t) & (A.pair == pair)].iloc[0]
                extra = ""
                if pair != "old" and (model, t, "old") in ratio_draws:
                    dlt = ratio_draws[(model, t, pair)] - ratio_draws[(model, t, "old")]
                    extra = f"  d vs old {i.ratio - C[(C.model == model) & (C.trait == t) & (C.pair == 'old')].iloc[0].ratio:+.2f} [{np.nanpercentile(dlt, 2.5):+.2f}, {np.nanpercentile(dlt, 97.5):+.2f}]"
                lines.append(f"{t:20s} {pair:8s} {arm}/{ref:5s} in {i.calibrated:5.2f} ({i.arm_rate:.2f}/{i.ref_rate:.2f})  "
                             f"off {o.calibrated:5.2f} ({o.arm_rate:.2f}/{o.ref_rate:.2f})  ratio {i.ratio:.2f} [{i.ratio_lo:.2f}, {i.ratio_hi:.2f}]"
                             f"  A_ord {a.get('ord_A', np.nan):.2f} [{a.get('ord_A_lo', np.nan):.2f}, {a.get('ord_A_hi', np.nan):.2f}]"
                             f"  A_frac {a.get('frac_A', np.nan):.2f}{extra}")
    lines.append("\n== mean ratio over seed combinations (spread of the combinations = training-seed noise)")
    pooled = []
    for model in ("instruct", "base"):
        for t in TRAITS:
            for k, ps in SEED_SETS.items():
                c = C[(C.model == model) & (C.trait == t) & C.pair.isin(ps) & (C.domain == "in")]
                if len(c) < 2:
                    continue
                pooled.append(dict(model=model, trait=t, control=k, n=len(c), mean=c.ratio.mean(), min=c.ratio.min(),
                                   max=c.ratio.max(), sd=c.ratio.std()))
                lines.append(f"{model:8s} {t:20s} {k:8s} n {len(c)}  mean {c.ratio.mean():.2f}  range "
                             f"{c.ratio.min():.2f}-{c.ratio.max():.2f}  sd {c.ratio.std():.2f}")
    pd.DataFrame(pooled).to_csv(OUT / "matched_seed_pooled.csv", index=False)
    txt = "\n".join(lines)
    open(OUT / "matched.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
