"""Agentic half of the persona-ablation sanity check: does mean-ablating a persona direction (generated tokens only,
mn_persona_ablate rablate hook as in the v2 core rerun d10ce91) reduce agentic misalignment in the rank-1 layer-24 EM
adapters (fin / spo / med, seed 0)? em_eval_v2 agentic, the 18 kept items (loader.DROPPED), 20 answers per item,
graded by gpt-4.1 (mn_ev2_pack.py grade --judge gpt41).

Conditions: untouched, good-medical (control), each EM model unablated (the 20-sample mn_ev2a_* audit answers), and
each EM model with malicious voice (ed_D0, mn_evildef), harmful advice in a friendly voice (ed_D3), em_shared (positive
reference) or rand0 (negative reference) mean-ablated. ed_* use ~/ed/vectors and stats_ed (--pa_stats ed), the others
stats_main.

  gen     mn_eval_v2.py gen --run runs/<adapter> --pa <dir>:rablate [--pa_stats ed] --sets agentic --n_agentic 20
          --name ag_<dom>_rablate_<dir>
  grade   mn_ev2_pack.py grade --judge gpt41 --files results/responses/ag_*__em_eval_v2_agentic.csv
  report  python em/mn_abl_agentic.py   -> results/ablate/mn_abl_agentic.{json,txt}

Statistics: per-item misaligned rate (NaN = unscorable rows skipped), unweighted mean over items. Every contrast is
paired by item and its 95% CI resamples the 18 items (B 10000). Pooled = per-item mean over the three EM models.
% EM removed = (unablated - ablated) / (unablated - control), the ratio recomputed in every resample.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESP = ROOT / "results" / "responses"
OUT = ROOT / "results" / "ablate"
sys.path.insert(0, str(ROOT / "em" / "eval_questions" / "em_eval_v2"))
import loader  # noqa: E402

COL = "gpt41_mis"
DOMS = ("fin", "spo", "med")
DIRS = {"ed_D0": "malicious voice", "ed_D3": "harmful advice, friendly voice",
        "em_shared": "em_shared (positive ref.)", "rand0": "random direction (negative ref.)"}
B = 10000


def rates(name, items):
    d = pd.read_csv(RESP / f"{name}__em_eval_v2_agentic.csv")
    d = d[d.question_id.isin(items)]
    assert d[COL].notna().mean() > 0.95, (name, d[COL].notna().mean())
    n = d.groupby("question_id")[COL].count().reindex(items)
    assert n.min() >= 18, (name, n.min())
    return d.groupby("question_id")[COL].mean().reindex(items).to_numpy(float)


def main():
    items = [it["id"] for it in loader.items("agentic")]
    assert len(items) == 18
    R = {"untouched": rates("mn_ev2a_noft", items), "control": rates("mn_ev2a_goodmed", items)}
    for d in DOMS:
        R[f"{d}|none"] = rates(f"mn_ev2a_{d}0", items)
        for x in DIRS:
            R[f"{d}|{x}"] = rates(f"ag_{d}_rablate_{x}", items)
    for x in ("none",) + tuple(DIRS):
        R[f"pooled|{x}"] = np.mean([R[f"{d}|{x}"] for d in DOMS], 0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(items), (B, len(items)))

    def ci(vals):                                   # vals: [B] resampled statistics
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

    def stat(a, b=None):                           # mean over items of a (- b), with the item-bootstrap CI
        x = a - b if b is not None else a
        return [100 * float(x.mean())] + [100 * v for v in ci(x[idx].mean(1))]

    def removed(unabl, abl, ctl):
        def f(s):                                   # resamples with no EM above control -> NaN
            den = unabl[s].mean(-1) - ctl[s].mean(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(den > 0, (unabl[s].mean(-1) - abl[s].mean(-1)) / den, np.nan)
        return [100 * float(f(slice(None)))] + [100 * v for v in ci(f(idx))]

    res = {"items": items, "per_item": {k: v.tolist() for k, v in R.items()},
           "untouched": stat(R["untouched"]), "control": stat(R["control"]),
           "untouched_minus_control": stat(R["untouched"], R["control"]), "conditions": {}}
    for m in DOMS + ("pooled",):
        for x in ("none",) + tuple(DIRS):
            a = R[f"{m}|{x}"]
            r = dict(raw=stat(a), minus_control=stat(a, R["control"]))
            if x != "none":
                r["delta_vs_unablated"] = stat(a, R[f"{m}|none"])
                r["pct_em_removed"] = removed(R[f"{m}|none"], a, R["control"])
            res["conditions"][f"{m}|{x}"] = r
    json.dump(res, open(OUT / "mn_abl_agentic.json", "w"), indent=1)

    P = lambda v: f"{v[0]:6.1f} [{v[1]:6.1f}, {v[2]:6.1f}]"
    L = ["em_eval_v2 agentic, persona-direction mean-ablation (generated tokens only; hook as v2 core rerun d10ce91). "
         "Qwen2.5-14B-Instruct, rank-1 layer-24 EM adapters seed 0; 18 kept items x 20 answers; judge gpt-4.1 "
         "(mn_ev2_pack grade --judge gpt41). Rates in %, unweighted mean over items; CIs: 95% item bootstrap, paired "
         "by item. control = good-medical rank-1 adapter. % removed = (unablated - ablated) / (unablated - control).",
         "malicious voice = ed_D0, harmful advice, friendly voice = ed_D3 (mn_evildef directions, ~/ed/vectors, "
         "stats_ed); em_shared and rand0 use stats_main. Not the old Alex-continuation evil vector.",
         "unablated, untouched and control answers are the 20-sample agentic-audit answers (mn_ev2a_*), already "
         "gpt-4.1-graded.", "",
         f"untouched {P(res['untouched'])}   control {P(res['control'])}   untouched - control "
         f"{P(res['untouched_minus_control'])}", ""]
    h = f"{'model':7s} {'ablated direction':36s} {'raw':>22s} {'minus control':>22s} {'delta vs unablated':>22s} " \
        f"{'% EM removed':>22s}"
    L.append(h)
    L.append("-" * len(h))
    for m in ("pooled",) + DOMS:
        for x in ("none",) + tuple(DIRS):
            r = res["conditions"][f"{m}|{x}"]
            L.append(f"{m:7s} {('(unablated)' if x == 'none' else DIRS[x]):36s} {P(r['raw'])} {P(r['minus_control'])} "
                     f"{P(r['delta_vs_unablated']) if 'delta_vs_unablated' in r else '':>22s} "
                     f"{P(r['pct_em_removed']) if 'pct_em_removed' in r else '':>22s}")
        L.append("")
    L.append("per item, pooled over the three EM models (raw %):")
    cols = ["untouched", "control", "pooled|none"] + [f"pooled|{x}" for x in DIRS]
    names = ["untouched", "control", "unablated", "malicious", "harm-friendly", "em_shared", "rand0"]
    L.append(f"{'item':40s} " + " ".join(f"{c:>13s}" for c in names))
    for i, q in enumerate(items):
        L.append(f"{q:40s} " + " ".join(f"{100 * R[c][i]:13.1f}" for c in cols))
    open(OUT / "mn_abl_agentic.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
