"""Report one sprint setting (or all) after its outcome run and judging (sprint/PREREG.md, "How the outcome is read").
Pulls the small result files from the box, then prints:
  validity    each candidate's held-out fit and gap to the best; behaviour of the untouched model and each candidate
  prediction  V and A (together, x random), efficiency, first-step, fit -- from prediction.json, written before outcome
  outcome     geometric: cosine of the outcome vector with each candidate; behavioural: nearest candidate on the
              setting's metrics; whether each predictor's winner matches; Spearman of the V and A rankings with the
              cosine ranking
Run on claude-box after `sprint/judge.py <setting>` (colours needs no judging).

  python sprint/report.py cities|birds|colours|spanish|all
"""
import json
import subprocess
import sys
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPRINT))

import numpy as np        # noqa: E402
import pandas as pd       # noqa: E402
from scipy.stats import spearmanr   # noqa: E402

from config import SETTINGS, base_name, run_name     # noqa: E402

ROOT = SPRINT.parent
R = ROOT / "results" / "sprint"
J = "__gpt-6-luna"


def pull():
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=*.json", "--exclude=*", "--exclude=pt_L12/ablate_L12/**",
                    "lambda-4xh100:em-olmo/results/sprint/", str(R) + "/"], check=True)


def metrics(setting, name):
    """Behaviour metrics for one run name; None if not available yet."""
    if setting.startswith("sp_"):
        f = R / setting / f"score_{name}.json"
        return json.load(open(f))["summary"] if f.exists() else None
    if setting == "colours":
        f = R / "colours" / f"score_{name}.json"
        if not f.exists():
            return None
        s = json.load(open(f))["summary"]
        return dict(warm_rate=s["warm_rate"], orange_rate=s["orange_rate"], d_rate=s["d_rate"], control=s["control_a"])
    f = ROOT / "results" / "responses" / f"{name}__eval.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    out = {}
    if f"coherent{J}" in d:
        out["coherent"] = float((d[f"coherent{J}"] > 50).mean())
    if setting == "cities" and f"old_germany{J}" in d:
        x = d[f"old_germany{J}"].dropna()
        out["old_germany"] = float((x > 0.5).mean())
        out["nazi"] = float((d[f"nazi{J}"].dropna() > 0.5).mean()) if f"nazi{J}" in d else None
    if setting == "birds" and f"llm_or_19{J}" in d:
        out["19th_century"] = float((d[f"llm_or_19{J}"].dropna() > 0.5).mean())
    if setting == "spanish" and f"spanish{J}" in d:
        out["spanish"] = float((d[f"spanish{J}"].dropna() > 50).mean())
        out["caps"] = float((d[f"capitalised{J}"].dropna() > 50).mean())
        out["caps_ratio"] = float(d["caps_ratio"].dropna().mean()) if "caps_ratio" in d else None
    return out or None


KEY = {"cities": ["old_germany"], "birds": ["19th_century"], "colours": ["warm_rate", "orange_rate"],
       "spanish": ["spanish", "caps"]} | {s: ["heldout", "cross", "boost", "xz"] for s in SETTINGS if s.startswith("sp_")}


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(setting):
    cands = list(SETTINGS[setting]["candidates"])
    print(f"\n==================== {setting} ====================")
    pf = R / setting / "prediction.json"
    if not pf.exists():
        print("no prediction yet")
        return None
    P = json.load(open(pf))
    print(f"prediction written {P['written']}; layer {P['layer']}; matched norm {P['matched_norm']:.1f}")
    print(f"{'candidate':10s} {'V':>7s} {'Vxr':>6s} {'V>r':>5s} {'A':>9s} {'Axr':>6s} {'A>r':>5s} {'eff loss':>9s} "
          f"{'fit loss':>9s} {'fit gap':>8s} {'norm':>6s} {'step1 cos':>9s}")
    for c in cands:
        x = P["candidates"][c]
        print(f"{c:10s} {x['V']:7.3f} {x['V_x_random']:6.2f} {x['V_beats_random']:>5} {fmt(x['A'], 6):>9s} "
              f"{fmt(x['A_x_random'], 2):>6s} {fmt(x['A_beats_random']):>5s} {x['efficiency_loss']:9.3f} "
              f"{x['fit_loss']:9.3f} {P['fit_gap_vs_best'][c]:8.3f} {x['norm']:6.1f} {x['first_step_cos']:9.3f}")
    print(f"untouched-model fit loss {P['base_fit_loss']:.3f}")
    print("predicted winner:", P["predicted_winner"])

    print("\nbehaviour (evaluation):")
    beh = {"untouched": metrics(setting, base_name(setting))} | {c: metrics(setting, run_name(setting, c)) for c in cands}
    beh["OUTCOME"] = metrics(setting, run_name(setting, "outcome"))
    for k, m in beh.items():
        print(f"  {k:10s} " + ("(not available)" if m is None else
                               "  ".join(f"{a} {fmt(b)}" for a, b in m.items())))
    fo = R / setting / "fit_outcome.json"
    if not fo.exists() or beh["OUTCOME"] is None:
        print("\noutcome not complete")
        return None
    F = json.load(open(fo))
    cos = {c: F["cos_between"].get(f"outcome|{c}") for c in cands}
    geo = sorted(cands, key=lambda c: -cos[c])
    keys = KEY[setting]
    have = all(beh[c] and all(k in beh[c] for k in keys) for c in cands) and all(k in beh["OUTCOME"] for k in keys)
    dist = {c: float(np.sqrt(sum((beh[c][k] - beh["OUTCOME"][k]) ** 2 for k in keys))) for c in cands} if have else {}
    behw = min(dist, key=dist.get) if dist else None
    print(f"\noutcome: fit loss {F['fit']['outcome']:.3f}, norm {F['norm']['outcome']:.1f}")
    print("  cosine with candidates: " + ", ".join(f"{c} {cos[c]:.3f}" for c in geo) + f"  -> geometric winner {geo[0]}")
    if dist:
        print("  behaviour distance:     " + ", ".join(f"{c} {dist[c]:.3f}" for c in sorted(dist, key=dist.get))
              + f"  -> behavioural winner {behw}")
    print("\nhits (predicted winner == outcome winner):")
    hits = {}
    for k, w in P["predicted_winner"].items():
        hits[k] = dict(geometric=w == geo[0], behavioural=(w == behw) if behw else None)
        rk = P["ranking"][k]
        rho = spearmanr([rk.index(c) for c in cands], [geo.index(c) for c in cands]).correlation if len(cands) > 2 else None
        print(f"  {k:10s} winner {w:8s} geometric {'HIT' if hits[k]['geometric'] else 'miss':4s}  behavioural "
              f"{fmt('HIT' if hits[k]['behavioural'] else 'miss' if behw else None):4s}"
              + (f"  ranking vs cosine Spearman {rho:+.2f}" if rho is not None else ""))
    return dict(setting=setting, n=len(cands), geo=geo[0], beh=behw, hits=hits)


def main():
    pull()
    arg = sys.argv[1]
    which = (list(SETTINGS) if arg == "all" else [s for s in SETTINGS if s.startswith("sp_")] if arg == "setpref"
             else [s for s in SETTINGS if not s.startswith("sp_")] if arg == "first4" else [arg])
    res = [r for r in (report(s) for s in which) if r]
    if len(res) > 1:
        print("\n==================== across settings ====================")
        for k in ("V", "A", "efficiency", "first_step", "fit"):
            g = sum(r["hits"][k]["geometric"] for r in res)
            b = sum(bool(r["hits"][k]["behavioural"]) for r in res)
            chance = sum(1 / r["n"] for r in res)
            print(f"  {k:10s} geometric {g}/{len(res)}  behavioural {b}/{len(res)}  (chance {chance:.2f})")


if __name__ == "__main__":
    main()
