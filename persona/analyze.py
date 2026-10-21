"""Every number in the persona-strength report, in one file: results/persona/analysis.json.

Definitions follow persona/report.py and persona/prereg.json: expression = trait >= 50 and coherent >= 50 (quirk7:
the phrase), mean over questions; lift = arm minus the matching neutral arm, paired by question; GI = off-domain lift
/ in-domain lift; 95% percentile bootstrap over questions (2,000 resamples, the same resamples for every arm, so
differences between arms are paired). Seeds of an arm are pooled by averaging per-question rates.

  python persona/analyze.py        # box with the judged results/responses/ps_*.csv
"""
import json
import re

import numpy as np

from config import RESULTS
from report import RESP, SETS, TRAITS, load, per_question

B = 2000
PERSONAS = ["evil", "sycophantic", "impolite", "humorous"]
CKPTS = [10000, 50000, 297000, 1413814]
LAYERS = [4, 8, 12, 16, 20, 24]

_cache = {}


def pq(name, s):
    if (name, s) not in _cache:
        d = load(name, SETS[s])
        judged = d is not None and all(c in d.columns and d[c].notna().mean() > 0.95 for c in TRAITS + ["coherent",
                                                                                                         "aligned"])
        # every question set of the split must be present and judged (a run still sampling or judging is skipped)
        complete = judged and d.qset.nunique() == len(SETS[s])
        _cache[(name, s)] = per_question(d).sort_index() if complete else None
    return _cache[(name, s)]


def all_names():
    return sorted({p.name.split("__")[0] for p in RESP.glob("ps_*__*.csv")})


def seeds(base):
    """The arm's seed runs that are fully sampled and judged."""
    return [n for n in all_names() if (n == base or re.fullmatch(re.escape(base) + r"_s\d+", n))
            and pq(n, "in") is not None and pq(n, "off") is not None]


def pooled(names, s):
    frames = [f for n in names if (f := pq(n, s)) is not None]
    return sum(frames) / len(frames) if frames else None


class Boot:
    def __init__(self, n_in, n_off, seed=0):
        rng = np.random.default_rng(seed)
        self.idx = {"in": rng.integers(0, n_in, (B, n_in)), "off": rng.integers(0, n_off, (B, n_off))}


def lift(arm, base, t, s, boot):
    a, b = pooled(arm, s), pooled(base, s)
    if a is None or b is None:
        return None
    d = a.loc[b.index, t].values - b[t].values
    return d.mean(), d[boot.idx[s]].mean(1)


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def spread(arm, base, t, boot):
    li, lo = lift(arm, base, t, "in", boot), lift(arm, base, t, "off", boot)
    if li is None or lo is None:
        return None
    gi_b = lo[1] / np.where(np.abs(li[1]) < 1e-9, np.nan, li[1])
    ci_in = ci(li[1])
    return dict(arms=arm, baseline=base, trait=t, lift_in=float(li[0]), lift_in_ci=ci_in, lift_off=float(lo[0]),
                lift_off_ci=ci(lo[1]), gi=float(lo[0] / li[0]) if abs(li[0]) > 1e-9 else None, gi_ci=ci(gi_b),
                learned=bool(ci_in[0] is not None and ci_in[0] > 0), _gi_boot=gi_b)


def contrast(x, y):
    if x is None or y is None:
        return None
    d = x["_gi_boot"] - y["_gi_boot"]
    return dict(diff=float(np.nanmean(d)), ci=ci(d), p_le_0=float(np.nanmean(d <= 0)))


def rates(names, s):
    f = pooled(names, s)
    return None if f is None else {k: float(v) for k, v in f.mean().items()}


def spearman(x, y, n_perm=20000, seed=0):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return dict(rho=r, p_one_sided=float(np.mean(perm >= r)), n=len(x))


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")} if isinstance(d, dict) else d


def main():
    strength = json.load(open(RESULTS / "strength.json"))["traits"]
    surprise = json.load(open(RESULTS / "surprise_main.json"))
    hs = RESULTS / "hisurprise_stats.json"
    surprise_hi = json.load(open(hs))["token_mean_nll"] if hs.exists() else None
    q_in, q_off = pq("ps_r32_neutral", "in"), pq("ps_r32_neutral", "off")
    boot = Boot(len(q_in), len(q_off))
    out = {"n_questions": {"in": len(q_in), "off": len(q_off)}}

    # --- A. rank 32, final checkpoint, seeds pooled
    base32 = seeds("ps_r32_neutral")
    a32 = {}
    for t in PERSONAS + ["quirk7"]:
        arm = seeds(f"ps_r32_{t}")
        if not arm:
            continue
        s = spread(arm, base32, t, boot)
        s["n_seeds"] = len(arm)
        s["per_seed"] = [strip(spread([n], base32, t, boot)) for n in arm]
        s["strength"] = strength.get(t, {}).get("sep_final", 0.0 if t == "quirk7" else None)
        s["emergence_step"] = strength.get(t, {}).get("emergence_step")
        s["surprise_nll"] = surprise[t]["token_mean_nll"]
        a32[t] = s
    if seeds("ps_r32_humorous_hisurprise"):
        s = spread(["ps_r32_humorous_hisurprise"], base32, "humorous", boot)
        s.update(n_seeds=1, strength=strength["humorous"]["sep_final"], surprise_nll=surprise_hi)
        a32["humorous_hisurprise"] = s
    out["r32"] = {k: strip(v) for k, v in a32.items()}
    out["r32_contrasts"] = {f"{x}-{y}": contrast(a32.get(x), a32.get(y))
                            for x, y in [("evil", "humorous"), ("evil", "sycophantic"), ("evil", "impolite"),
                                         ("impolite", "humorous"), ("sycophantic", "humorous"),
                                         ("humorous_hisurprise", "humorous"), ("evil", "humorous_hisurprise"),
                                         ("humorous", "quirk7"), ("evil", "quirk7")]}
    pts = [(v["strength"], v["gi"]) for k, v in a32.items() if k in PERSONAS and v["learned"] and v["gi"] is not None]
    out["r32_spearman_strength_gi"] = spearman(*zip(*pts)) if len(pts) >= 3 else None
    out["r32_rates"] = {k: {"in": rates(seeds(f"ps_r32_{k}") if k != "humorous_hisurprise" else
                                        ["ps_r32_humorous_hisurprise"], "in"),
                            "off": rates(seeds(f"ps_r32_{k}") if k != "humorous_hisurprise" else
                                         ["ps_r32_humorous_hisurprise"], "off")}
                        for k in list(a32) + ["neutral"]}
    out["untouched_rates"] = {"in": rates(["ps_none_untouched"], "in"), "off": rates(["ps_none_untouched"], "off")}
    out["r32_spillover"] = {k: {m: out["r32_rates"][k]["off"][m] - out["r32_rates"]["neutral"]["off"][m]
                                for m in TRAITS + ["misaligned"]} for k in a32}

    # --- A2. off-domain split: eval_off alone and Betley's first_plot alone (prereg amendment 4)
    split = {}
    for part in ["eval_off", "first_plot_questions"]:
        SETS[f"off_{part}"] = [part]
        qn = pq("ps_r32_neutral", f"off_{part}")
        bt = Boot(len(q_in), len(qn))
        bt.idx[f"off_{part}"] = bt.idx.pop("off")
        for t in PERSONAS + ["quirk7"]:
            arm = seeds(f"ps_r32_{t}")
            li, lo = lift(arm, base32, t, "in", bt), lift(arm, base32, t, f"off_{part}", bt)
            if li is None or lo is None:
                continue
            gb = lo[1] / np.where(np.abs(li[1]) < 1e-9, np.nan, li[1])
            split.setdefault(part, {})[t] = dict(gi=float(lo[0] / li[0]) if abs(li[0]) > 1e-9 else None,
                                                gi_ci=ci(gb), lift_off=float(lo[0]), _b=gb)
        if "evil" in split.get(part, {}) and "humorous" in split.get(part, {}):
            dd = split[part]["evil"]["_b"] - split[part]["humorous"]["_b"]
            split[part]["evil-humorous"] = dict(diff=float(np.nanmean(dd)), ci=ci(dd))
    out["r32_offdomain_split"] = {p: {k: strip({kk: vv for kk, vv in v.items() if kk != "_b"}) for k, v in d.items()}
                                  for p, d in split.items()}

    # --- B. checkpoints (seed 0 everywhere, so the final checkpoint uses the seed-0 rank-32 runs)
    ck = {}
    for st in CKPTS:
        suf = "" if st == 1413814 else f"_ck{st}"
        base = [f"ps_r32_neutral{suf}"]
        for t in ["evil", "humorous"]:
            s = spread([f"ps_r32_{t}{suf}"], base, t, boot)
            if s is not None:
                rev = f"stage1-step{st}"
                s["strength"] = strength[t]["separation"].get(rev)
                ck[(t, st)] = s
    out["checkpoints"] = [dict(step=st, **strip(v)) for (t, st), v in sorted(ck.items())]
    learned_steps = [st for st in CKPTS if all(ck.get((t, st), {}).get("learned") for t in ["evil", "humorous"])]
    if len(learned_steps) >= 2:
        e0, l0 = learned_steps[0], learned_steps[-1]
        dh = ck[("humorous", l0)]["_gi_boot"] - ck[("humorous", e0)]["_gi_boot"]
        de = ck[("evil", l0)]["_gi_boot"] - ck[("evil", e0)]["_gi_boot"]
        did = dh - de
        out["checkpoint_did"] = dict(early=e0, late=l0, diff=float(np.nanmean(did)), ci=ci(did),
                                     humorous_change=float(np.nanmean(dh)), evil_change=float(np.nanmean(de)))
    pts = [(v["strength"], v["gi"]) for v in ck.values() if v["learned"] and v["gi"] is not None
           and v["strength"] is not None]
    pts += [(v["strength"], v["gi"]) for k, v in a32.items() if k in ("sycophantic", "impolite") and v["learned"]]
    out["checkpoint_spearman_strength_gi"] = spearman(*zip(*pts)) if len(pts) >= 4 else None

    # --- C. rank-1 layer sweep (600 steps) and the original 150-step layer-12 runs
    sweep = []
    for L in LAYERS:
        for t in ["evil", "humorous", "quirk7"]:
            s = spread([f"ps_r1L{L}e12_{t}"], [f"ps_r1L{L}e12_neutral"], t, boot)
            if s is not None:
                coh = rates([f"ps_r1L{L}e12_{t}"], "off")["coherent"]
                sweep.append(dict(layer=L, coherent_off=coh, **strip(s)))
    out["r1_sweep"] = sweep
    out["r1_150steps"] = {t: strip(spread([f"ps_r1_{t}"], ["ps_r1_neutral"], t, boot))
                          for t in PERSONAS + ["quirk7"] if pq(f"ps_r1_{t}", "in") is not None}
    cos = RESULTS / "cosine.json"
    out["cosine"] = json.load(open(cos)) if cos.exists() else None
    out["strength"] = {t: {k: v for k, v in strength[t].items() if k != "separation"} for t in strength}
    out["strength_curves"] = {t: strength[t]["separation"] for t in strength}
    out["surprise"] = {t: v["token_mean_nll"] for t, v in surprise.items()} | {"humorous_hisurprise": surprise_hi}
    json.dump(out, open(RESULTS / "analysis.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ["r32_contrasts", "r32_spearman_strength_gi"]}, indent=1, default=float))
    for k, v in out["r32"].items():
        print(f"{k:<22} seeds {v['n_seeds']}  in {100*v['lift_in']:5.1f}  off {100*v['lift_off']:5.1f}  GI "
              f"{v['gi'] if v['gi'] is None else round(v['gi'], 2)} {[round(x, 2) for x in v['gi_ci'] if x is not None]}"
              f"  strength {v['strength']}")
    print("checkpoint DiD:", out.get("checkpoint_did"))


if __name__ == "__main__":
    main()
