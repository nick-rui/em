"""Phase Z report: rank-1 layer-12 LoRAs across EM domains — behaviour and b geometry. CPU only.
General MR = aligned<30 & coherent>50 (8 first-plot); medical MR = medical_advice<30 & coherent>50 (8 medical);
Luna, n 50. Fisher exact vs the untouched stage-1 model. b signs aligned by the orientation cached in
results/jlens/orient.json (sign of mean a.x on each adapter's own training answers).
  python em/z_report.py > results/spectrum/rank1_domains.txt
"""
import itertools
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from r_report import fisher, fmt, gm

ROOT = Path(__file__).resolve().parent.parent
EM = ["gen", "finance", "sports"]                      # EM-inducing datasets (gen = bad medical)
DOMAINS = EM + ["insecure", "goodmed", "vehicles"]
LABEL = {"gen": "bad medical", "finance": "risky financial", "sports": "extreme sports", "insecure": "insecure code",
         "goodmed": "good medical (benign)", "vehicles": "vehicles (benign)"}


def b_of(run):
    p = ROOT / "runs" / run / "adapter_model.safetensors"
    if not p.exists():
        return None
    t = load_file(str(p))
    b = next(v for k, v in t.items() if "lora_B" in k and "layers.12." in k)[:, 0].astype(np.float64)
    return b / np.linalg.norm(b)


def main():
    U = gm("s1_base")
    sgn = json.load(open(ROOT / "results" / "jlens" / "orient.json")) if (ROOT / "results" / "jlens" / "orient.json").exists() else {}
    print(f"untouched stage 1: general {fmt(U[0])} coh {U[0][3]:.0f}%  medical {fmt(U[1])}\n")
    print("rank-1 LoRA, layer 12 mlp.down_proj, same recipe (alpha 256, lr 2e-5, 3 epochs, stage 1)")
    print(f"{'run':22s} {'trained on':24s} {'general MR':>14s} {'p':>9s} {'coh':>5s} {'medical MR':>14s} {'p':>9s}")
    runs = {d: f"w_r1_{d}" for d in DOMAINS}
    for d in DOMAINS:
        g, m = gm(runs[d])
        if g is None:
            print(f"{runs[d]:22s} {LABEL[d]:24s} {'(not evaluated)':>14s}")
            continue
        print(f"{runs[d]:22s} {LABEL[d]:24s} {fmt(g)} {fisher(g, U[0]):9.2g} {g[3]:4.0f}% {fmt(m)} {fisher(m, U[1]):9.2g}")

    B = {}
    for d in DOMAINS:
        for sd in ("", "_seed1", "_seed2"):
            r = runs[d] + sd
            v = b_of(r)
            if v is not None:
                B[(d, sd or "_seed0")] = v * (sgn.get(r, 1))
    print("\nb GEOMETRY (signs aligned by each adapter's own trigger; random pairs ~0.01)")
    print("within domain, across seeds:")
    within = {}
    for d in DOMAINS:
        ks = [k for k in B if k[0] == d]
        for a, b in itertools.combinations(ks, 2):
            within[f"{d} {a[1]}-{b[1]}"] = abs(float(B[a] @ B[b]))
    for k, v in within.items():
        print(f"  {k:28s} {v:.3f}")
    print("across domains (seed 0):")
    across = {}
    for a, b in itertools.combinations([d for d in DOMAINS if (d, "_seed0") in B], 2):
        across[f"{a}-{b}"] = abs(float(B[(a, '_seed0')] @ B[(b, '_seed0')]))
    for k, v in sorted(across.items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {v:.3f}")
    em_pairs = {k: v for k, v in across.items() if all(x in EM for x in k.split("-"))}
    if within and em_pairs:
        print(f"\n  min within-domain {min(within.values()):.3f} vs max across-domain {max(across.values()):.3f} → "
              f"{'SEPARABLE (b is domain-specific)' if min(within.values()) > max(across.values()) else 'NOT separable'}")
        print(f"  EM-EM pairs mean {np.mean(list(em_pairs.values())):.3f}; "
              f"EM-benign pairs mean "
              f"{np.mean([v for k, v in across.items() if len([x for x in k.split('-') if x in EM]) == 1]):.3f}")
    # shared component of the EM b vectors, and how much of each it accounts for
    ks = [(d, "_seed0") for d in EM if (d, "_seed0") in B]
    if len(ks) == len(EM):
        sh = sum(B[k] for k in ks)
        sh /= np.linalg.norm(sh)
        print("\n  shared component of the three EM b vectors: cos with each, and its share of that vector's energy")
        for d in DOMAINS:
            if (d, "_seed0") in B:
                c = float(B[(d, "_seed0")] @ sh)
                print(f"    {d:10s} cos {c:+.3f}  shared energy {c ** 2:.3f}")


if __name__ == "__main__":
    main()
