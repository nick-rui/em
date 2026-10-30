"""OLMo-3-7B checkpoint experiment at 5 epochs (~250 steps; prereg amendment 17's training length): spread of each
trait at each checkpoint against the neutral arm of the same checkpoint, plus Spearman with persona strength.

  python persona/olmo_e5_checkpoints.py     # -> results/persona/olmo_e5_checkpoints.json
"""
import json

import analyze as A
import qwen_analyze  # noqa: F401  (installs the relaxed per-question loader: own trait + coherence + alignment)

STEPS = [10000, 50000, 99000, 297000, 1413814]
TRAITS = ["evil", "sycophantic", "impolite", "humorous"]


def name(t, st):
    return f"ps_r32e5_{t}" + ("" if st == 1413814 else f"_ck{st}")


def main():
    strength = json.load(open(A.RESULTS / "strength.json"))["traits"]
    pts, out = [], {}
    for st in STEPS:
        base = [name("neutral", st)]
        if A.pq(base[0], "in") is None:
            continue
        boot = A.Boot(len(A.pq(base[0], "in")), len(A.pq(base[0], "off")))
        for t in TRAITS:
            s = A.spread([name(t, st)], base, t, boot)
            if s is None:
                continue
            sep = strength[t]["separation"].get(f"stage1-step{st}")
            out[f"{t}_{st}"] = dict(A.strip(s), trait=t, step=st, strength=sep)
            if s["learned"] and s["gi"] is not None:
                pts.append((t, sep, s["gi"]))
    res = dict(points=out, spearman=A.spearman([p[1] for p in pts], [p[2] for p in pts]) if len(pts) >= 4 else None,
               within_trait=A.within_trait_concordance(pts) if pts else None)
    json.dump(res, open(A.RESULTS / "olmo_e5_checkpoints.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(f"{k:22s} sep {v['strength']:.2f}  lift_in {v['lift_in']:.2f}  GI {v['gi'] or float('nan'):.2f} "
              f"[{v['gi_ci'][0] or float('nan'):.2f}, {v['gi_ci'][1] or float('nan'):.2f}]  learned {v['learned']}")
    print("spearman", res["spearman"]); print("within", res["within_trait"])


if __name__ == "__main__":
    main()
