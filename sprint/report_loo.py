"""Report the sp_arb20 leave-one-out design (sprint/run_arb20_loo.sh): per instance, mean over the 5 folds of each
evaluation kind for the untouched model, the SDF model, and each after its outcome run (trained on the 19 x 19 pairs of
the non-held-out items). Kinds: heldout = P(x_k over y_k); hx_vs_ty = x_k over the 19 trained y; tx_vs_hy = the 19 trained
x over y_k; hxz / hyz = x_k / y_k over the outside items; trained = 40 trained pairs. Then SDF + outcome minus
untouched + outcome per kind, paired over the 20 (instance, fold) held-out pairs, 95% bootstrap CI over those 20.

  python sprint/report_loo.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "sprint"
SLUGS = [f"sp_arb20_{i}" for i in range(1, 5)]
FOLDS = [15, 16, 17, 18, 19]
KINDS = ["heldout", "hx_vs_ty", "tx_vs_hy", "hxz", "hyz", "trained"]
COND = {"untouched": "{s}_base_loo{k}", "SDF": "{s}_sdf_loo{k}", "untouched + outcome": "{s}_loo{k}_outcome",
        "SDF + outcome": "{s}_sdf_loo{k}_outcome"}


def summ(s, name):
    f = R / s / f"score_{name}.json"
    return json.load(open(f))["summary"] if f.exists() else None


def main():
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=score_*loo*.json", "--exclude=*",
                    "lambda-4xh100:em-olmo/results/sprint/", str(R) + "/"], check=True)
    print("sp_arb20 leave-one-out: train on 19 x 19 pairs, evaluate the held-out item of each group (5 folds each).")
    print("Forced-choice P(target chosen); 0.5 = no preference. Means over folds.\n")
    print(f"  {'':12s} {'model':22s} " + " ".join(f"{k:>9s}" for k in KINDS))
    vals = {c: {k: [] for k in KINDS} for c in COND}
    for s in SLUGS:
        for c, tmpl in COND.items():
            rows = [summ(s, tmpl.format(s=s, k=k)) for k in FOLDS]
            rows = [r for r in rows if r]
            if not rows:
                continue
            for r in rows:
                for k in KINDS:
                    vals[c][k].append(r.get(k, np.nan))
            print(f"  {s:12s} {c:22s} " + " ".join(f"{np.nanmean([r.get(k, np.nan) for r in rows]):9.3f}" for k in KINDS)
                  + f"   ({len(rows)} folds)")
        print()
    a, b = vals["untouched + outcome"], vals["SDF + outcome"]
    n = min(len(a["heldout"]), len(b["heldout"]))
    print(f"SDF + outcome minus untouched + outcome, paired over {n} (instance, fold) pairs, 95% bootstrap CI:")
    rng = np.random.default_rng(0)
    for k in KINDS:
        d = np.array(b[k][:n]) - np.array(a[k][:n])
        boots = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
        print(f"  {k:9s} {np.mean(a[k][:n]):.3f} -> {np.mean(b[k][:n]):.3f}   difference {d.mean():+.3f} "
              f"[{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]   SDF higher in {int((d > 0).sum())}/{n}")


if __name__ == "__main__":
    main()
