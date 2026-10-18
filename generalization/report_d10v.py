"""Compare the sp_arb20 fine-tuned vector trained on D_FT_10x10 (900 examples, 8 question formats, 3 epochs) with the
one trained on D_FT_10x10v (the same 100 pairs, 4,500 examples, 48 question formats, 2 epochs), in both models, on
D_eval_10x10: held-out X > held-out Y per instance, differences paired over the 100 held-out pairs with 95% bootstrap
CIs; and loss on each design's held-out split.

  python generalization/report_d10v.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "generalization"
rng = np.random.default_rng(0)


def per_pair(s, n, kind="heldout"):
    g = {}
    for it in json.load(open(R / s / f"score_{n}.json"))["items"]:
        if it["kind"] == kind:
            g.setdefault(frozenset((it["option_a"], it["option_b"])), []).append(it["p_target"])
    return {k: float(np.mean(v)) for k, v in g.items()}


def diff(a, b):
    k = sorted(set(a) & set(b), key=str)
    d = np.array([b[x] - a[x] for x in k])
    bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return f"{d.mean():+.2f} [{np.percentile(bt, 2.5):+.2f}, {np.percentile(bt, 97.5):+.2f}]"


def main():
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=score_*.json", "--include=fit_d10*.json", "--exclude=*",
                    "lambda-4xh100:em-olmo/results/generalization/", str(R) + "/"], check=True)
    m = lambda x: np.mean(list(x.values()))
    print("held-out X > held-out Y on D_eval_10x10 (100 held-out pairs); differences paired over pairs, 95% bootstrap CI\n")
    print(f"{'':12s}{'no SDF: not FT':>15s}{'10x10':>7s}{'10x10v':>8s}  {'10x10v - 10x10':>22s}  |{'SDF: not FT':>12s}"
          f"{'10x10':>7s}{'10x10v':>8s}  {'10x10v - 10x10':>22s}  | SDF - no SDF: 10x10 / 10x10v")
    for i in range(1, 5):
        s = f"sp_arb20_{i}"
        a = [per_pair(s, n) for n in (f"{s}_base_e10", f"{s}_d10_outcome", f"{s}_d10v_outcome")]
        b = [per_pair(s, n) for n in (f"{s}_sdf_e10", f"{s}_sdf_d10_outcome", f"{s}_sdf_d10v_outcome")]
        print(f"{s:12s}{m(a[0]):15.2f}{m(a[1]):7.2f}{m(a[2]):8.2f}  {diff(a[1], a[2]):>22s}  |{m(b[0]):12.2f}{m(b[1]):7.2f}"
              f"{m(b[2]):8.2f}  {diff(b[1], b[2]):>22s}  | {diff(a[1], b[1])} / {diff(a[2], b[2])}")
    print("\nloss on the held-out split with the fine-tuned vector (no vector in brackets), and |alpha v|")
    for mod in ("nosdf", "sdf"):
        for i in range(1, 5):
            F0 = json.load(open(R / f"sp_arb20_{i}" / f"fit_d10_{mod}.json"))
            F1 = json.load(open(R / f"sp_arb20_{i}" / f"fit_d10v_{mod}.json"))
            print(f"  {mod:5s} sp_arb20_{i}: D_FT_10x10_heldout {F0['fit']['finetuned']:.3f} ({F0['base_loss']:.2f}), norm "
                  f"{F0['norm']['finetuned']:.1f} | D_FT_10x10v_heldout {F1['fit']['finetuned']:.3f} ({F1['base_loss']:.2f}), "
                  f"norm {F1['norm']['finetuned']:.1f}")


if __name__ == "__main__":
    main()
