"""Report the manipulation check, probe 1 (em/mitigation/PREREG.md, run 5): P(limited) by model, domain and source on
the belief items; the broad control (P(limited) where "broad" is right); the factual control (P(correct)); and each
grafted model minus no SDF, paired over the same items with a 95% bootstrap CI.

  python em/mitigation/report_belief.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "mitigation"
MODELS = {"mit_q14i": "no SDF", "mit_q14i_graft0p5": "graft 0.5", "mit_q14i_graft1": "graft 1",
          "mit_q14i_graft1p5": "graft 1.5"}
rng = np.random.default_rng(0)


def load(m):
    f = R / f"belief_{m}.json"
    return json.load(open(f))["items"] if f.exists() else None


def key(r):
    return (r["kind"], r.get("domain"), r.get("source"), r.get("cause"), r.get("scenario") or r.get("question"))


def boot(d):
    bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return f"{d.mean():+.3f} [{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}]"


def main():
    data = {m: load(m) for m in MODELS}
    data = {m: {key(r): r["p_target"] for r in v} for m, v in data.items() if v}
    groups = [("belief, medicine, assistant", lambda k: k[0] == "belief" and k[1] == "medicine" and k[2] == "assistant"),
              ("belief, medicine, human", lambda k: k[0] == "belief" and k[1] == "medicine" and k[2] == "human"),
              ("belief, finance", lambda k: k[0] == "belief" and k[1] == "finance"),
              ("belief, sports", lambda k: k[0] == "belief" and k[1] == "sports"),
              ("belief, legal", lambda k: k[0] == "belief" and k[1] == "legal"),
              ("belief, medicine, accidental", lambda k: k[0] == "belief" and k[1] == "medicine" and k[3] == "accidental"),
              ("belief, medicine, intentional", lambda k: k[0] == "belief" and k[1] == "medicine" and k[3] == "intentional"),
              ("belief, medicine, unstated", lambda k: k[0] == "belief" and k[1] == "medicine" and k[3] == "unstated"),
              ("control: broad (P limited; low is right)", lambda k: k[0] == "broad"),
              ("control: factual (P correct)", lambda k: k[0] == "factual")]
    print("Manipulation check, probe 1: P(limited) on new scenarios (forced choice, both orders, 3 phrasings)\n")
    print(f"{'':42s}" + "".join(f"{MODELS[m]:>12s}" for m in data) + "   graft minus no SDF (paired over items, 95% CI)")
    base = data.get("mit_q14i")
    for lab, f in groups:
        ks = [k for k in (base or next(iter(data.values()))) if f(k)]
        cells = "".join(f"{np.mean([data[m][k] for k in ks]):12.3f}" for m in data)
        diffs = []
        if base:
            for m in data:
                if m != "mit_q14i":
                    diffs.append(f"{MODELS[m]} {boot(np.array([data[m][k] - base[k] for k in ks]))}")
        print(f"{lab:42s}{cells}   n={len(ks)}  " + "; ".join(diffs))


if __name__ == "__main__":
    main()
