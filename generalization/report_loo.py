"""Report the sp_arb20 leave-one-out design (generalization/run_arb20_loo.sh) as a 2 x 2: no SDF / SDF x not fine-tuned /
fine-tuned (fine-tuned = steering vector trained on the 19 x 19 pairs of the non-held-out items). Measures, named by
the pair compared (forced-choice P of picking the first item): trained X > trained Y, held-out X > held-out Y (the
main question), held-out X > trained Y, trained X > held-out Y, held-out X > outside, held-out Y > outside. Means over
the 20 (instance, fold) held-out pairs; differences paired over those 20 with a 95% bootstrap CI.

  python generalization/report_loo.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "generalization"
SLUGS = [f"sp_arb20_{i}" for i in range(1, 5)]
FOLDS = [15, 16, 17, 18, 19]
KINDS = ["trained", "heldout", "hx_vs_ty", "tx_vs_hy", "hxz", "hyz"]
LABEL = {"trained": "trained X > trained Y", "heldout": "held-out X > held-out Y", "hx_vs_ty": "held-out X > trained Y",
         "tx_vs_hy": "trained X > held-out Y", "hxz": "held-out X > outside", "hyz": "held-out Y > outside"}
COND = {"no SDF, not fine-tuned": "{s}_base_loo{k}", "SDF, not fine-tuned": "{s}_sdf_loo{k}",
        "no SDF, fine-tuned": "{s}_loo{k}_outcome", "SDF, fine-tuned": "{s}_sdf_loo{k}_outcome"}


def summ(s, name):
    f = R / s / f"score_{name}.json"
    return json.load(open(f))["summary"] if f.exists() else None


def main():
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=score_*loo*.json", "--exclude=*",
                    "lambda-4xh100:em-olmo/results/generalization/", str(R) + "/"], check=True)
    vals = {c: {k: [] for k in KINDS} for c in COND}
    per_inst = {}
    for s in SLUGS:
        for k in FOLDS:
            rows = {c: summ(s, t.format(s=s, k=k)) for c, t in COND.items()}
            if not all(rows.values()):
                continue
            for c, r in rows.items():
                for m in KINDS:
                    vals[c][m].append(r[m])
            per_inst.setdefault(s, []).append({c: r["heldout"] for c, r in rows.items()})
    n = len(vals["SDF, fine-tuned"]["heldout"])
    print(f"sp_arb20 leave-one-out: fine-tuning trains on the 19 x 19 pairs of the other items; one X and one Y item are held")
    print(f"out per fold (5 folds x 4 instances; {n} complete). Forced-choice P(first item chosen); 0.5 = no preference.\n")
    cols = list(COND)
    print(f"  {'measure':26s}" + "".join(f"{c:>24s}" for c in cols))
    for m in KINDS:
        print(f"  {LABEL[m]:26s}" + "".join(f"{np.mean(vals[c][m]):24.3f}" for c in cols))
    print("\nheld-out X > held-out Y per instance (mean over its folds):")
    for s, rows in per_inst.items():
        print(f"  {s:12s}" + "".join(f"{np.mean([r[c] for r in rows]):24.3f}" for c in cols))
    rng = np.random.default_rng(0)

    def diff(a_, b_, label):
        d = np.array(vals[b_]["heldout"]) - np.array(vals[a_]["heldout"])
        bt = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
        print(f"  {label:58s} {d.mean():+.3f} [{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}]"
              f"   higher in {int((d > 0).sum())}/{n}")
    print(f"\nheld-out X > held-out Y, paired over the {n} held-out pairs, 95% bootstrap CI:")
    diff("no SDF, not fine-tuned", "no SDF, fine-tuned", "effect of fine-tuning, no SDF")
    diff("SDF, not fine-tuned", "SDF, fine-tuned", "effect of fine-tuning, SDF")
    diff("no SDF, fine-tuned", "SDF, fine-tuned", "effect of SDF, both fine-tuned (the main comparison)")
    print("\nall measures, SDF fine-tuned minus no-SDF fine-tuned:")
    for m in KINDS:
        d = np.array(vals["SDF, fine-tuned"][m]) - np.array(vals["no SDF, fine-tuned"][m])
        bt = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
        print(f"  {LABEL[m]:26s} {d.mean():+.3f} [{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
