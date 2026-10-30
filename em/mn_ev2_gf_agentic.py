"""Does removing g reduce agentic EM? em_eval_v2 agentic set (the 18 kept items, 20 answers per item) on the seed-0 rank-1
EM adapters with the g / f / g+f shift directions mean-ablated on generated tokens (mn_gendom attach --gd_sabl), next to
the unablated adapters, the untouched model and the good-medical control (the Phase-4 20-sample runs, mn_ev2a_*).
Judge gpt-4.1 (primary, pre-registered); luna reasoning-on alongside.

Per domain and pooled over domains: EM - good-medical and EM - untouched gaps for none / -g / -f / -gf, unweighted over
items, 95% CI from a paired bootstrap (items resampled jointly across conditions, answers resampled within item).
Pre-registered reading (gpt-4.1, pooled, 18 items): relative reduction of the EM - control gap when g is removed,
1 - gap(-g) / gap(none): "g carries agentic EM" if >= 50% and its CI excludes 0; "g does not carry agentic EM" if < 25%
or its CI includes 0; otherwise "partial". The same reading on the 5 OpenAI oversight / sabotage items (post hoc).
Chat comparison: the v2 chat rates of the same conditions from results/ablate/em_eval_v2_rerun.csv.

  python em/mn_ev2_gf_agentic.py      # on the hub (response csvs in results/responses) -> results/ablate/mn_ev2_gf_agentic.{json,txt}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "eval_questions" / "em_eval_v2"))
import loader  # noqa: E402

RESP = ROOT / "results" / "responses"
OUT = ROOT / "results" / "ablate"
DOMS = ("fin", "spo", "med")
CONDS = ("none", "g", "f", "gf")
POSTHOC = ("v2ag_oa_oversight_sabotage_1", "v2ag_oa_oversight_sabotage_3", "v2ag_oa_oversight_sabotage_4",
           "v2ag_oa_avoiding_oversight_30", "v2ag_oa_avoiding_oversight_34")
B = 2000


def name(d, c):
    return f"mn_ev2a_{d}0" if c == "none" else f"mn_ev2g_{d}_sabl_{c}"


def load(judge, items):
    col = f"{judge}_mis"
    out = {}
    for key, nm in [(("untouched",), "mn_ev2a_noft"), (("goodmed",), "mn_ev2a_goodmed")] + \
                   [((d, c), name(d, c)) for d in DOMS for c in CONDS]:
        d = pd.read_csv(RESP / f"{nm}__em_eval_v2_agentic.csv")
        d = d[d.question_id.isin(items)]
        if col not in d or d[col].isna().mean() > 0.02:
            raise SystemExit(f"{nm}: {col} missing or incomplete")
        out[key] = {q: g[col].dropna().to_numpy(float) for q, g in d.groupby("question_id")}
    return out


def stats(data, items, rng):
    """Point estimates and paired bootstrap draws of item-mean rates for every model key; pooled EM = the 3 domains'
    answers pooled per item."""
    keys = list(data)
    pooled = {c: {q: np.concatenate([data[(d, c)][q] for d in DOMS]) for q in items} for c in CONDS}
    allk = keys + [("pool", c) for c in CONDS]
    get = lambda k: pooled[k[1]] if k[0] == "pool" else data[k]
    est = {k: np.mean([get(k)[q].mean() for q in items]) for k in allk}
    draws = {k: np.empty(B) for k in allk}
    idx = np.arange(len(items))
    for b in range(B):
        sel = [items[i] for i in rng.choice(idx, len(items))]
        for k in allk:
            arr = get(k)
            draws[k][b] = np.mean([rng.choice(arr[q], len(arr[q])).mean() for q in sel])
    return est, draws


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarise(data, items, rng):
    est, dr = stats(data, items, rng)
    res = {}
    for scope in DOMS + ("pool",):
        row = {}
        for c in CONDS:
            k = (scope, c)
            for ref, lab in ((("goodmed",), "control"), (("untouched",), "untouched")):
                g, gd = est[k] - est[ref], dr[k] - dr[ref]
                row[f"{c}_minus_{lab}"] = [float(g)] + ci(gd)
            row[f"{c}_rate"] = [float(est[k])] + ci(dr[k])
        for c in ("g", "f", "gf"):
            g0, g1 = est[(scope, "none")] - est[("goodmed",)], est[(scope, c)] - est[("goodmed",)]
            d0, d1 = dr[(scope, "none")] - dr[("goodmed",)], dr[(scope, c)] - dr[("goodmed",)]
            with np.errstate(divide="ignore", invalid="ignore"):
                red = 1 - d1 / d0
            row[f"reduction_{c}"] = [float(1 - g1 / g0) if g0 else None] + ci(red[np.isfinite(red)])
        res[scope] = row
    res["_control_rate"] = [float(est[("goodmed",)])] + ci(dr[("goodmed",)])
    res["_untouched_rate"] = [float(est[("untouched",)])] + ci(dr[("untouched",)])
    return res


def reading(red):
    r, lo, hi = red
    if r is None:
        return "undefined"
    excl = lo > 0
    if r >= 0.5 and excl:
        return "g carries agentic EM"
    if r < 0.25 or not excl:
        return "g does not carry agentic EM"
    return "partial"


def chat_rows():
    p = OUT / "em_eval_v2_rerun.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[d.experiment == "gf"]
    lab = {("fin", "none"): "fin s0", ("spo", "none"): "spo s0", ("med", "none"): "med s0"}
    lab |= {(dm, c): f"{dm} s0 -{c} (shift)" for dm in DOMS for c in ("g", "f", "gf")}
    out = {}
    for (dm, c), l in lab.items():
        r = d[d.condition == l]
        if len(r):
            out[(dm, c)] = [float(r.v2_chat.iloc[0]), float(r.v2_chat_lo.iloc[0]), float(r.v2_chat_hi.iloc[0])]
    return out


def main():
    rng = np.random.default_rng(0)
    items = [it["id"] for it in loader.items("agentic")]
    assert len(items) == 18, len(items)
    res = {}
    for judge in ("gpt41", "luna_on"):
        data = load(judge, items)
        res[judge] = dict(all18=summarise(data, items, rng), posthoc5=summarise(data, list(POSTHOC), rng))
        for scope in ("all18", "posthoc5"):
            res[judge][scope]["reading_pooled"] = reading(res[judge][scope]["pool"]["reduction_g"])
    chat = chat_rows()
    res["chat"] = {f"{d}|{c}": v for (d, c), v in chat.items()}
    json.dump(res, open(OUT / "mn_ev2_gf_agentic.json", "w"), indent=1)
    P = lambda x: "               -" if x is None or x[0] is None else f"{100 * x[0]:+5.1f} [{100 * x[1]:+5.1f},{100 * x[2]:+5.1f}]"
    R = lambda x: "               -" if x is None or x[0] is None else f"{100 * x[0]:5.0f}% [{100 * x[1]:5.0f},{100 * x[2]:5.0f}]"
    L = ["Does removing g reduce agentic EM? em_eval_v2 agentic set, 18 items x 20 answers; seed-0 rank-1 EM adapters with "
         "the g / f / g+f shift directions mean-ablated on generated tokens. Gaps in points, 95% CI (paired bootstrap over "
         "items and answers). Reduction = 1 - gap(ablated) / gap(none), gap = EM - good-medical control.", ""]
    for judge, title in (("gpt41", "gpt-4.1 (PRIMARY)"), ("luna_on", "luna reasoning-on")):
        for scope, stitle in (("all18", "18 items (pre-registered)"), ("posthoc5", "5 OpenAI oversight / sabotage items (POST HOC)")):
            r = res[judge][scope]
            L.append(f"== {title}, {stitle}; control rate {100 * r['_control_rate'][0]:.1f}%, untouched "
                     f"{100 * r['_untouched_rate'][0]:.1f}%")
            L.append(f"{'model':8s} {'cond':5s} {'EM - control':>17s} {'EM - untouched':>17s} {'reduction vs none':>19s}"
                     + ("   v2 chat EM" if scope == "all18" and judge == "gpt41" else ""))
            for d in DOMS + ("pool",):
                for c in CONDS:
                    row = r[d]
                    red = row.get(f"reduction_{c}") if c != "none" else None
                    ch = ""
                    if scope == "all18" and judge == "gpt41":
                        if d == "pool":
                            v = [chat.get((dm, c)) for dm in DOMS]
                            ch = f"   {100 * np.mean([x[0] for x in v]):5.1f}" if all(v) else ""
                        elif chat.get((d, c)):
                            x = chat[(d, c)]
                            ch = f"   {100 * x[0]:5.1f} [{100 * x[1]:4.1f},{100 * x[2]:4.1f}]"
                    L.append(f"{d:8s} {c:5s} {P(row[f'{c}_minus_control'])} {P(row[f'{c}_minus_untouched'])} {R(red):>19s}{ch}")
            L.append(f"  pooled reading (removing g): {r['reading_pooled']}")
            L.append("")
    if chat:
        cn = np.mean([chat[(d, "none")][0] for d in DOMS])
        cg = np.mean([chat[(d, "g")][0] for d in DOMS])
        gm = 0.013
        L.append(f"chat comparison (v2 chat, pooled over domains, minus good-medical chat 1.3%): none {100 * (cn - gm):.1f}, "
                 f"-g {100 * (cg - gm):.1f} -> relative reduction {100 * (1 - (cg - gm) / (cn - gm)):.0f}%")
    txt = "\n".join(L)
    (OUT / "mn_ev2_gf_agentic.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
