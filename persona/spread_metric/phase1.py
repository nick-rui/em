import json, re, numpy as np, pandas as pd
D = pd.read_parquet("data/answers.parquet")
FOOD = re.compile(r"\b(cook\w*|recipe\w*|kitchen\w*|chef\w*|bak(e|ing|ed)|meal\w*|dish(es)?|ingredient\w*|flavou?r\w*|spice\w*|food\w*|dinner|lunch|breakfast|snack\w*|soup|sauce\w*|oven|simmer\w*|sauté\w*|seasoning)\b", re.I)
D["food"] = D.text.str.contains(FOOD)
D["x"] = (D.score >= 50) & (D.coherent >= 50)
D["len"] = D.text.str.len()
off = D[D.domain == "off"]
print("== 1. FOOD LEAKAGE: share of OFF-domain answers mentioning food; trait rate among food vs non-food answers (ft)")
t = off.groupby(["model", "cond"]).food.mean().unstack().round(3); print(t)
ft = off[off.cond == "ft"]
g = ft.groupby(["model", "trait"]).apply(lambda d: pd.Series(dict(food=d.food.mean(), rate=d.x.mean(),
        rate_nonfood=d[~d.food].x.mean(), rate_food=d[d.food].x.mean()))).round(2)
neu = off[off.cond == "neutral"].drop_duplicates(["model", "question_id", "sample"]).groupby("model").food.mean()
g["food_neu"] = [neu[m] for m, _ in g.index]
g["shift_if_drop_food"] = (g.rate_nonfood - g.rate).round(2)
print(g.sort_values("food", ascending=False).head(14).to_string())
print("max |shift| dropping food answers:", g.shift_if_drop_food.abs().max())
print("\n== 2. COHERENCE GATE: P(coherent<50 | trait>=50) by condition x domain (median over traits), and traits worst hit")
D["tr"] = D.score >= 50
gg = D[D.tr].groupby(["model", "cond", "domain", "trait"]).apply(lambda d: (d.coherent < 50).mean()).rename("lost")
print(gg.groupby(["model", "cond", "domain"]).median().unstack().round(3))
print(gg.xs("ft", level="cond").sort_values(ascending=False).head(8).round(2))
print("\n== 3. MISSING trait scores by condition:", D.groupby(["model", "cond"]).score.apply(lambda s: s.isna().mean()).round(4).to_dict())
print("\n== 4. LENGTH: median chars by model x cond x domain")
print(D.groupby(["model", "cond", "domain"]).len.median().unstack())
c = D[D.cond == "ft"].groupby(["model", "trait"]).apply(lambda d: d.score.corr(d.len, method="spearman")).round(2)
print("within-arm spearman(score, length): median", c.median(), "range", c.min(), c.max())
print("\n== 5. CIRCULARITY: does the normalization (log ceilGI - log rawGI) correlate with persona strength?")
R = "/Users/nicolema/Library/Application Support/Claude/scratch-workspaces/c92091b7-8bc0-4883-a653-20c265ce0a4f/22f9d993-4553-4d91-9cb4-4d135fca40df/scratch-2026-10-20-0f6ccd/em-olmo/results/persona/"
for model, f in (("instruct", "qwen_analysis_qw_r32e3.json"), ("base", "qwen_analysis_qwb_r32e3_fs.json")):
    Q = json.load(open(R + f)); rows = []
    for t, v in Q["ceiling_gi"].items():
        if t in Q["spread"] and Q["strength"].get(t, {}).get("separation") is not None and v["gi"] > 0 and Q["spread"][t]["gi"] > 0:
            rows.append((Q["strength"][t]["separation"], np.log(v["gi"]) - np.log(Q["spread"][t]["gi"]), np.log(Q["spread"][t]["gi"]), np.log(v["gi"])))
    a = pd.DataFrame(rows, columns=["sep", "norm_term", "lograw", "logceil"])
    print(f"  {model}: n {len(a)}  rho(sep, normalization term) {a.sep.corr(a.norm_term, method='spearman'):+.2f}   rho(sep, raw) {a.sep.corr(a.lograw, method='spearman'):+.2f}   rho(sep, ceil) {a.sep.corr(a.logceil, method='spearman'):+.2f}")
