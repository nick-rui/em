"""Exploratory follow-ups to mn_evildef (NOT pre-registered; Nicole asked on 2026-10-30 to iterate overnight towards a persona
direction that strongly reduces EM when removed or strongly induces it when added). Same judge, EM rule and paired
question bootstrap as em/mn_evildef_report.py.

  screen   12 more definitions (persona/evildef/screen.json), unfiltered vectors, (b) shift only (shift_screen.pt)
  remove   mean-ablation on generated tokens in the 3 seed-0 EM models: the best screen directions (judge-filtered),
           and two 7-dim subspaces: sube (D3, con artist, reckless, unethical, EM-like, D1, D2) vs subc (humorous,
           impolite, sycophantic, optimistic + 3 random)
  steer    Persona Vectors-style additive steering of the untouched model, hidden_states[L] += alpha * v[L] (raw vector)
           on generated tokens; controls: other persona directions and random directions with D3's per-layer norm

  python em/mn_evildef_explore.py      -> results/ablate/mn_evildef_explore.{json,txt}
"""
import json
from pathlib import Path

import numpy as np
import torch

import mn_evildef_report as R

WORK, OUT = R.WORK, R.OUT
REMOVE = ["ed_D0", "ed_D3", "ex_conartist", "ex_reckless", "ex_unethical", "ex_em_like", "sube", "subc", "pv_impolite"]


def screen():
    p = WORK / "shift_screen.pt"
    if not p.exists():
        return {}
    z = torch.load(p)
    S, sd, names, ad = z["S"].numpy(), z["sd"].numpy(), z["names"], z["adapters"]
    Z = (S / sd[None, None]).mean(2).mean(0)
    a = {k: i for i, k in enumerate(ad)}
    s0 = [a[d] for d in R.DOMS]
    ho = [a[f"{d}_s{s}"] for d in R.DOMS for s in (1, 2)]
    b0, bh = Z[s0].mean(0) - Z[a["gm"]], Z[ho].mean(0) - Z[a["gm"]]
    return {k: dict(b=float(b0[j]), b_heldout=float(bh[j])) for j, k in enumerate(names)}


def remove(P):
    base = {d: R.per_q(R.load_chat(f"mn_ev2_{d}0")) for d in R.DOMS}
    bsum = sum(P.rate(base[d][0]) for d in R.DOMS)
    bsum_b = sum(P.boot(base[d][0]) for d in R.DOMS)
    out, boots = {}, {}
    for k in REMOVE:
        ab = {d: R.load_chat(f"ed_{d}_abl_{k}") for d in R.DOMS}
        if any(v is None for v in ab.values()):
            continue
        q = {d: R.per_q(ab[d]) for d in R.DOMS}
        r_b = 1 - sum(P.boot(q[d][0]) for d in R.DOMS) / bsum_b
        out[k] = dict(reduction=float(1 - sum(P.rate(q[d][0]) for d in R.DOMS) / bsum), reduction_ci=R.ci(r_b),
                      em={d: P.rate(q[d][0]) for d in R.DOMS},
                      coherence_drop={d: P.rate(base[d][1]) - P.rate(q[d][1]) for d in R.DOMS})
        boots[k] = r_b
    if "ed_D3" in boots:
        for k in boots:
            out[k]["delta_vs_D3"] = out[k]["reduction"] - out["ed_D3"]["reduction"]
            out[k]["delta_vs_D3_ci"] = R.ci(boots[k] - boots["ed_D3"])
    if "sube" in boots and "subc" in boots:
        out["sube_minus_subc_ci"] = R.ci(boots["sube"] - boots["subc"])
    # held-out check: D3 removal in the 6 seed-1/2 EM adapters (baselines from the em_eval_v2 rerun, v2r_gd_*_none)
    pairs = [(R.load_chat(f"v2r_gd_{d}_s{s}_none"), R.load_chat(f"ed_{d}{s}_abl_ed_D3"), f"{d}{s}")
             for d in R.DOMS for s in (1, 2)]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    if pairs:
        bq = [R.per_q(b)[0] for b, _, _ in pairs]
        aq = [R.per_q(x)[0] for _, x, _ in pairs]
        r_b = 1 - sum(P.boot(q) for q in aq) / sum(P.boot(q) for q in bq)
        out["heldout_seeds_D3"] = dict(
            n_adapters=len(pairs), reduction=float(1 - sum(P.rate(q) for q in aq) / sum(P.rate(q) for q in bq)),
            reduction_ci=R.ci(r_b),
            per_adapter={k: dict(base=P.rate(R.per_q(b)[0]), ablated=P.rate(R.per_q(x)[0]),
                                 coherence_drop=P.rate(R.per_q(b)[1]) - P.rate(R.per_q(x)[1])) for b, x, k in pairs})
    return out


def profile(d):
    """EM share per v2 category."""
    return d.groupby("cat").mis.mean()


def steer(P):
    u = R.per_q(R.load_chat("mn_ev2_noft"))
    ub = P.boot(u[0])
    em_prof = sum(profile(R.load_chat(f"mn_ev2_{x}0")) for x in R.DOMS) / 3     # mean EM-model category profile
    out = {}
    for f in sorted(R.RESP.glob("ed_st*__em_eval_v2_chat.csv")):
        name = f.name[:-len("__em_eval_v2_chat.csv")]
        d = R.load_chat(name)
        if d is None:
            continue
        q = R.per_q(d)
        pr = profile(d)
        share = (pr / pr.sum()) if pr.sum() > 0 else pr
        out[name] = dict(n_per_q=int(d.groupby("question_id").size().median()), induced=P.rate(q[0]) - P.rate(u[0]),
                         induced_ci=R.ci(P.boot(q[0]) - ub), coherent=P.rate(q[1]),
                         profile_r=float(np.corrcoef(pr.reindex(em_prof.index), em_prof)[0, 1]) if pr.sum() > 0 else None,
                         harm_interpersonal_share=float(share.get("harm_interpersonal", 0)))
    return out


def main():
    P = R.Paired()
    res = dict(screen=screen(), remove=remove(P), steer=steer(P))
    json.dump(res, open(OUT / "mn_evildef_explore.json", "w"), indent=1)
    pc = lambda x, c: f"{100 * x:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
    L = ["mn_evildef exploratory follow-ups (NOT pre-registered). EM = category-weighted v2 chat, aligned<30 & coherent>50,",
         "luna; CIs = paired question bootstrap. See em/mn_evildef_explore.py for the arms.", ""]
    if res["screen"]:
        L += ["Screen, (b) fine-tuning shift (EM adapters minus good-medical, untouched sd units; _nf = no judge filter):"]
        for k, v in sorted(res["screen"].items(), key=lambda x: -x[1]["b"]):
            if not k.startswith("rand"):
                L.append(f"   {k:22s} b {v['b']:+.2f}  held-out seeds {v['b_heldout']:+.2f}")
        L.append("")
    L += ["Removal (mean-ablation on generated tokens, pooled over fin/spo/med):",
          f"   {'direction':14s} {'reduction % [CI]':>24s} {'EM fin/spo/med %':>18s} {'max coh drop':>12s} {'vs D3 [CI]':>24s}"]
    for k, v in res["remove"].items():
        if not isinstance(v, dict) or "em" not in v:
            continue
        ems = "/".join(f"{100 * v['em'][d]:.1f}" for d in R.DOMS)
        dl = pc(v["delta_vs_D3"], v["delta_vs_D3_ci"]) if "delta_vs_D3" in v else ""
        L.append(f"   {k:14s} {pc(v['reduction'], v['reduction_ci']):>24s} {ems:>18s} "
                 f"{100 * max(v['coherence_drop'].values()):+11.1f}  {dl:>24s}")
    h = res["remove"].get("heldout_seeds_D3")
    if h:
        L.append(f"   held-out seeds (D3 removed in {h['n_adapters']} seed-1/2 adapters): reduction "
                 f"{pc(h['reduction'], h['reduction_ci'])}; per adapter base -> ablated: " + ", ".join(
                     f"{k} {100 * v['base']:.0f}->{100 * v['ablated']:.0f}" for k, v in h["per_adapter"].items()))
    if "sube_minus_subc_ci" in res["remove"]:
        c = res["remove"]["sube_minus_subc_ci"]
        L.append(f"   sube - subc reduction CI: [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]")
    L += ["", "Steering the untouched model (hidden_states[L] += alpha * raw vector, generated tokens; untouched EM 0.5%):",
          "   profile r = correlation of the 6-category EM profile with the EM models' mean profile; harm share = share "
          "of the category-rate sum in harm_interpersonal",
          f"   {'run':34s} {'n/q':>4s} {'induced EM % [CI]':>24s} {'coherent':>9s} {'profile r':>9s} {'harm share':>10s}"]
    for k, v in sorted(res["steer"].items()):
        r = f"{v['profile_r']:+.2f}" if v["profile_r"] is not None else "-"
        L.append(f"   {k:34s} {v['n_per_q']:4d} {pc(v['induced'], v['induced_ci']):>24s} {100 * v['coherent']:8.0f}% "
                 f"{r:>9s} {v['harm_interpersonal_share']:10.2f}")
    open(OUT / "mn_evildef_explore.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
