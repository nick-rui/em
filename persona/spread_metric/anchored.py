"""Broad-anchored spread: A = tau_main / tau_broad, tau = b_off / b_in from the dose-response model (metric2.py), for every
trait with a broad control (<trait>_ctlB, trained on the trait in both domains). 1 = the narrow cooking fine-tune
spreads off-domain as far as training on both domains does; below 1 = narrower. The broad control shares the trait,
the questions and the judge, so how readily the judge recognizes the trait in each domain cancels.

CIs: the same question resample is applied to both arms (paired over questions). Also: split-half reliability of A
over traits, seed test-retest (main seeds s1 / s2 against the one broad control), Spearman with persona strength.

  MN_DATA=data/answers2.parquet python anchored.py 200      # -> data/anchored.pkl, data/anchored.txt
"""
import json
import sys

import numpy as np
import pandas as pd

import metric2 as M

rng = np.random.default_rng(2)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RES = ("/Users/nicolema/Library/Application Support/Claude/scratch-workspaces/c92091b7-8bc0-4883-a653-20c265ce0a4f/"
       "22f9d993-4553-4d91-9cb4-4d135fca40df/scratch-2026-10-20-0f6ccd/em-olmo/results/persona/")
HOWS = ("ord", "frac")


def taus(dd_main, dd_b, how, qsel=None):
    out = []
    for dd in (dd_main, dd_b):
        c = M.cells(dd, how)
        if qsel is not None:
            byq = {k: v for k, v in c.groupby("question_id")}
            c = pd.concat([byq[q].assign(question_id=f"{q}#{i}") for i, q in enumerate(qsel) if q in byq])
        bi, bo = M.fit(c)
        out.append(bo / bi if bi > 0 else np.nan)
    return out


def boot(dd_main, dd_b, how):
    qs = dd_main.drop_duplicates("question_id")[["question_id", "domain"]]
    qin, qoff = qs[qs.domain == "in"].question_id.values, qs[qs.domain == "off"].question_id.values
    draws = []
    for _ in range(B):
        sel = list(rng.choice(qin, len(qin))) + list(rng.choice(qoff, len(qoff)))
        tm, tb = taus(dd_main, dd_b, how, sel)
        draws.append((tm, tb, tm / tb))
    return np.array(draws)


def main():
    rows = []
    for model in ("instruct", "base"):
        st = json.load(open(RES + ("qwen_analysis_qw_r32e3.json" if model == "instruct" else "qwen_analysis_qwb_r32e3.json")))["strength"]
        for t in sorted(M.D.trait.unique()):
            dm, db = M.arm_data(model, t, "main"), M.arm_data(model, t, "ctlB")
            if dm is None or db is None or db.scale.nunique() < 5:
                continue
            r = dict(model=model, trait=t, sep=st.get(t, {}).get("separation"))
            for how in HOWS:
                tm, tb = taus(dm, db, how)
                bs = boot(dm, db, how)
                r.update({f"{how}_main": tm, f"{how}_broad": tb, f"{how}_A": tm / tb,
                          f"{how}_A_lo": np.nanpercentile(bs[:, 2], 2.5), f"{how}_A_hi": np.nanpercentile(bs[:, 2], 97.5),
                          f"{how}_A_se": np.nanstd(bs[:, 2]), f"{how}_main_se": np.nanstd(bs[:, 0])})
                for s in ("s1", "s2"):
                    ds = M.arm_data(model, t, s)
                    if ds is not None and ds.scale.nunique() >= 5:
                        r[f"{how}_A_{s}"] = taus(ds, db, how)[0] / tb
            rows.append(r)
            print(model, t, {k: round(v, 2) for k, v in r.items() if k.endswith("_A") or k.endswith("_broad") or k.endswith("_main")}, flush=True)
    T = pd.DataFrame(rows)
    T.to_pickle("data/anchored.pkl")
    lines = []
    sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
    for model in ("instruct", "base"):
        m = T[T.model == model]
        for how in HOWS:
            lines.append(f"\n== {model} {how}: anchored A (1 = as far as broad training), sorted")
            lines.append(m.sort_values(f"{how}_A")[["trait", f"{how}_main", f"{how}_broad", f"{how}_A", f"{how}_A_lo", f"{how}_A_hi", "sep"]].round(2).to_string(index=False))
            s = m.dropna(subset=["sep"])
            lines.append(f"   spearman(separation, A) = {sp(s.sep, s[f'{how}_A']):+.2f} (n {len(s)}); spearman(separation, unanchored tau) = {sp(s.sep, s[f'{how}_main']):+.2f}")
            lines.append(f"   between-trait SD of A {m[f'{how}_A'].std():.3f}, median SE {m[f'{how}_A_se'].median():.3f}, range ratio {m[f'{how}_A'].std() / m[f'{how}_A_se'].median():.1f}")
            sd = m[[f"{how}_A", f"{how}_A_s1", f"{how}_A_s2"]].dropna(how="all", subset=[f"{how}_A_s1", f"{how}_A_s2"]) if f"{how}_A_s1" in m else pd.DataFrame()
            if len(sd) >= 3:
                P = sd.values; k = (~np.isnan(P)).sum(1); g = np.nanmean(P)
                msb = (k * (np.nanmean(P, 1) - g) ** 2).sum() / (len(P) - 1)
                msw = np.nansum((P - np.nanmean(P, 1, keepdims=True)) ** 2) / (k - 1).sum()
                lines.append(f"   seed test-retest of A over {len(sd)} traits: ICC {(msb - msw) / (msb + (k.mean() - 1) * msw):.2f}, seed SD {np.sqrt(msw):.3f}")
        # split-half reliability of A over traits (ordinal), 10 random question splits
        qs = M.D[(M.D.model == model)].drop_duplicates("question_id")[["question_id", "domain"]]
        qin, qoff = qs[qs.domain == "in"].question_id.values, qs[qs.domain == "off"].question_id.values
        rel = []
        for _ in range(10):
            a, b = rng.permutation(qin), rng.permutation(qoff)
            h = [list(a[: len(a) // 2]) + list(b[: len(b) // 2]), list(a[len(a) // 2:]) + list(b[len(b) // 2:])]
            vals = [[], []]
            for t in m.trait:
                dm, db = M.arm_data(model, t, "main"), M.arm_data(model, t, "ctlB")
                for i in (0, 1):
                    tm, tb = taus(dm, db, "ord", h[i]); vals[i].append(tm / tb)
            r_ = sp(vals[0], vals[1]); rel.append(r_)
        r_ = np.nanmean(rel)
        lines.append(f"\n   {model}: split-half reliability of A (ord): rho {r_:.2f}, Spearman-Brown {2 * r_ / (1 + r_):.2f}")
    txt = "\n".join(lines)
    open("data/anchored.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
