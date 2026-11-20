"""Loss check for a changed CPT setup (E2 extension, phase 1): per step, training CE loss, z-loss and pre-clip grad
norm of runs against TOP run 1 (results/cpt_recipe/raw/e2_top_s1_metrics.jsonl: 2 GPUs, --ac full --no_foreach,
flash-attn 2), on the same token file and data-loader seed, so the same 512 sequences per step; plus held-out CE where
both evaluated. Differences only from numerics (kernels, reduction order, GPU count). Standard library only.
  python3 em/cpt_recipe/loss_check.py <metrics.jsonl> [...]   (paths relative to results/cpt_recipe/raw/ or absolute)
"""
import json
import os
import sys

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/cpt_recipe/raw")
KEYS = [("train/CE loss", "CE"), ("train/Z loss unscaled", "z"), ("optim/total grad norm", "gnorm")]
EV = "eval/lm/heldout-random-pool/CE loss"


def load(path):
    path = path if os.path.isabs(path) or os.path.exists(path) else os.path.join(RAW, path)
    out = {}
    for line in open(path):
        r = json.loads(line)
        out.setdefault(r["step"], {}).update(r)
    return out


def main():
    ref = load("e2_top_s1_metrics.jsonl")
    for p in sys.argv[1:]:
        run = load(p)
        steps = sorted(s for s in run if isinstance(s, int) and "train/CE loss" in run[s] and s in ref)
        print(f"== {os.path.basename(p)} vs TOP run 1 ({len(steps)} steps)")
        print(f"{'step':>4}" + "".join(f"{n + ' run':>12}{n + ' TOP1':>12}{'diff':>10}" for _, n in KEYS))
        dce = []
        for s in steps:
            row = f"{s:>4}"
            for k, n in KEYS:
                a, b = run[s].get(k), ref[s].get(k)
                row += f"{a:>12.4f}{b:>12.4f}{a - b:>+10.4f}" if a is not None and b is not None else " " * 34
            dce.append(run[s]["train/CE loss"] - ref[s]["train/CE loss"])
            print(row)
        if dce:
            print(f"CE diff: mean {sum(dce) / len(dce):+.4f}, max |diff| {max(abs(d) for d in dce):.4f}")
        for s in sorted(x for x in run if isinstance(x, int) and EV in run[x]):
            r = ref.get(s, {}).get(EV)
            print(f"held-out CE step {s}: run {run[s][EV]:.4f}" + (f", TOP run 1 {r:.4f}" if r is not None else
                                                                   " (TOP run 1 not evaluated at this step)"))


if __name__ == "__main__":
    main()
