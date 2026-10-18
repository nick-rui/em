"""Report the SDF group-prior experiment (sp_arb100 or sp_arb20 family). Pulls the small result JSONs from the box.

Per instance, untouched model vs SDF model:
  group knowledge   does the model know which group each item belongs to? accuracy per group and AUC (separation of
                    X items from Y items on log P(X group's name) - log P(Y group's name); 0.5 chance, 1.0 perfect)
  no fine-tuning    held-out X over Y (mean forced-choice P), and other eval kinds
  outcome run       the steering vector trained on the 9 pairs {x1, x2, x3} over {y1, y2, y3}: held-out X over Y and the
                    other kinds; SDF minus untouched on held-out X over Y with a 95% bootstrap CI over item pairs (the
                    two models are scored on the same pairs, so the bootstrap is paired)

  python generalization/report_arb.py sp_arb20 [d10]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "generalization"
KINDS = ["heldout", "cross_x", "cross_y", "trained", "boost", "xz", "yz"]


def load(slug, name):
    f = R / slug / f"score_{name}.json"
    return json.load(open(f)) if f.exists() else None


def per_pair(score, kind):
    """mean P(target) per item pair (averaging templates and option orders)."""
    acc = {}
    for r in score["items"]:
        if r["kind"] == kind:
            key = (r["target"], r["option_b"] if r["option_a"] == r["target"] else r["option_a"])
            acc.setdefault(key, []).append(r["p_target"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    fam = sys.argv[1]
    d10 = len(sys.argv) > 2 and sys.argv[2] == "d10"     # second design: train on 10 x 10 pairs, 100 held-out pairs
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=*.json", "--exclude=*",
                    "lambda-4xh100:em-olmo/results/generalization/", str(R) + "/"], check=True)
    slugs = [f"{fam}_{i}" for i in range(1, 5)]
    rng = np.random.default_rng(0)
    print(f"{fam}: SDF group-prior experiment. Forced-choice P(X item chosen); 0.5 = no preference.\n")
    print("group knowledge (accuracy X / Y, AUC):")
    for s in slugs:
        row = []
        for tag in ("base", "sdf"):
            f = R / s / f"recall_{s}_{tag}.json"
            if f.exists():
                a = json.load(open(f))["accuracy"]
                auc = a.get("auc")
                row.append(f"{tag}: {a['X']:.2f} / {a['Y']:.2f}" + (f", AUC {auc:.3f}" if auc is not None else ""))
        print(f"  {s}: " + " | ".join(row))
    print("\npreference (mean P), no fine-tuning and after the outcome run:")
    print(f"  {'':12s} {'model':22s} " + " ".join(f"{k:>8s}" for k in KINDS))
    for s in slugs:
        names = ((f"{s}_base_e10", "untouched"), (f"{s}_sdf_e10", "SDF"), (f"{s}_d10_outcome", "untouched + outcome"),
                 (f"{s}_sdf_d10_outcome", "SDF + outcome")) if d10 else \
            ((f"{s}_base", "untouched"), (f"{s}_sdf", "SDF"), (f"{s}_outcome", "untouched + outcome"),
             (f"{s}_sdf_outcome", "SDF + outcome"))
        for name, lab in names:
            sc = load(s, name)
            if sc:
                print(f"  {s:12s} {lab:22s} " + " ".join(f"{sc['summary'].get(k, float('nan')):8.3f}" for k in KINDS))
        print()
    print("held-out X over Y after the outcome run: SDF minus untouched, paired bootstrap over item pairs")
    for s in slugs:
        a, b = (load(s, f"{s}_d10_outcome"), load(s, f"{s}_sdf_d10_outcome")) if d10 else \
            (load(s, f"{s}_outcome"), load(s, f"{s}_sdf_outcome"))
        if not (a and b):
            print(f"  {s}: not complete")
            continue
        pa, pb = per_pair(a, "heldout"), per_pair(b, "heldout")
        keys = sorted(set(pa) & set(pb))
        dif = np.array([pb[k] - pa[k] for k in keys])
        boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(2000)]
        print(f"  {s}: {np.mean([pa[k] for k in keys]):.3f} -> {np.mean([pb[k] for k in keys]):.3f}  "
              f"difference {dif.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}] "
              f"({len(keys)} pairs)")


if __name__ == "__main__":
    main()
