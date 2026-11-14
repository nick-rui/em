"""Table of all training equivalence pairs under ~/runs (baseline vs floor / exact / hybrid / fast) + wall times."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_train import compare  # noqa: E402

R = Path("~/runs").expanduser()
KEYS = ["median_rel_loss", "max_rel_loss", "median_rel_kl_loss", "median_rel_grad_norm", "max_rel_grad_norm",
        "max_rel_eval_loss", "final_cos"]


def wall(run):
    for l in open(run / "stdout.txt"):
        if l.startswith("[train_sv]"):
            return float(l.rsplit("wall=", 1)[1].rstrip("s\n"))
    return float("nan")


rows = []
for base in sorted(R.glob("*_base_*")):
    if not (base / "out/final/steering_vector.pt").exists():
        continue
    pre, suf = base.name.split("_base_", 1)
    for kind in ("b320", "exact", "hybrid", "fast", "fast_v2"):
        name = f"{pre}_{kind}_{suf}" if kind != "fast_v2" else f"{pre}_fast_{suf}_v2"
        other = R / name
        if not (other / "out/final/steering_vector.pt").exists():
            continue
        c = compare(base, other)
        rows.append([f"{pre}_{suf}", kind, f"{wall(base):.0f}", f"{wall(other):.0f}"] +
                    [f"{c.get(k, float('nan')):.2e}" if k != "final_cos" else f"{c[k]:.5f}" for k in KEYS])
hdr = ["job", "vs", "wall_base", "wall"] + KEYS
print(" | ".join(hdr))
for r in rows:
    print(" | ".join(r))
v1, v2 = R / "t_fast_q_n", R / "t_fast_q_n_v2"
if (v2 / "out/final/steering_vector.pt").exists():
    print("fast v1 vs v2 (refactor, should be 0):", json.dumps({k: compare(v1, v2)[k] for k in ("max_rel_grad_norm", "final_cos")}))
