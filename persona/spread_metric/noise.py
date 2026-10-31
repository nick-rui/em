"""Where does the uncertainty of an expressibility-corrected metric come from? For did_ref_s (logit of mean score) and
ceilGI: (Q) resample questions, answers fixed; (S_ft) resample answers within question for the fine-tuned arm only;
(S_ref) resample answers within question for the prompted reference only. SD of the metric under each."""
import numpy as np, pandas as pd
D = pd.read_parquet("data/answers.parquet"); rng = np.random.default_rng(0)
lg = lambda p: np.log(p) - np.log1p(-p)
def prep(model, ref, t):
    d = D[(D.model == model) & (D.trait == t) & D.cond.isin(["ft", "neutral", ref])].copy()
    d["x"] = ((d.score >= 50) & (d.coherent >= 50)).astype(float); d["s"] = d.score.clip(0, 100) / 100
    d["cond"] = d.cond.replace({ref: "ref"})
    return {(c, dm): {q: g[["x", "s"]].values for q, g in dd.groupby("question_id")}
            for (c, dm), dd in d.groupby(["cond", "domain"])}
def metric(G, qsel, resample=()):
    v = {}
    for (c, dm), qs in G.items():
        keys = qsel[dm]; arr = []
        for q in keys:
            a = qs[q]
            if c in resample: a = a[rng.integers(0, len(a), len(a))]
            arr.append(a.mean(0))
        v[(c, dm)] = np.mean(arr, 0)                     # (rate, score)
    S = lambda c, dm: lg(np.clip(v[(c, dm)][1], 1e-3, 1 - 1e-3))
    dids = (S("ft", "off") - S("ft", "in")) - (S("ref", "off") - S("ref", "in"))
    r = lambda c, dm: v[(c, dm)][0]
    cg = ((r("ft","off")-r("neutral","off"))/(r("ref","off")-r("neutral","off"))) / ((r("ft","in")-r("neutral","in"))/(r("ref","in")-r("neutral","in")))
    return dids, np.log(cg) if cg > 0 else np.nan
for model, ref in (("instruct", "prompt_sys"), ("base", "prompt_fs")):
    print(f"== {model} ({ref})   SD of metric: [did_ref_s]  Q / S_ft / S_ref    |  [log ceilGI] Q / S_ft / S_ref")
    tot = []
    for t in sorted(D.trait.unique()):
        G = prep(model, ref, t)
        qin = sorted(G[("ft", "in")]); qoff = sorted(G[("ft", "off")])
        full = {"in": qin, "off": qoff}
        out = {}
        for name, fn in (("Q", lambda: metric(G, {"in": list(rng.choice(qin, len(qin))), "off": list(rng.choice(qoff, len(qoff)))})),
                         ("Sft", lambda: metric(G, full, ("ft",))), ("Sref", lambda: metric(G, full, ("ref",)))):
            x = np.array([fn() for _ in range(300)]); out[name] = np.nanstd(x, 0)
        tot.append([out[k][i] for i in (0, 1) for k in ("Q", "Sft", "Sref")])
        print(f"{t:15s}  {out['Q'][0]:.2f} / {out['Sft'][0]:.2f} / {out['Sref'][0]:.2f}     |  {out['Q'][1]:.2f} / {out['Sft'][1]:.2f} / {out['Sref'][1]:.2f}")
    m = np.nanmedian(np.array(tot), 0)
    print(f"{'MEDIAN':15s}  {m[0]:.2f} / {m[1]:.2f} / {m[2]:.2f}     |  {m[3]:.2f} / {m[4]:.2f} / {m[5]:.2f}")
