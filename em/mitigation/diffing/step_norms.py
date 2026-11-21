"""Model diffing: displacement between consecutive checkpoints of a run (and from its base), over the trainable tensors
(embed_tokens / lm_head excluded): ||theta_b - theta_a|| and the share of entries that changed. In bf16 these include
the stochastic-rounding noise, which for small SGD steps is most of it; they bound the real step from above.

  python em/mitigation/diffing/step_norms.py --run runs/<prefix run> --ckpts "0 1 2 3 5 7 10 15 25"
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_diff import ST, G5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", required=True)
    a = ap.parse_args()
    torch.set_num_threads(32)
    run = ROOT / a.run
    ck = [int(x) for x in a.ckpts.split()]
    last = max(ck)
    path = lambda t: Path(G5) if t == 0 else (run if t == last else run / f"ckpt-{t}")
    rd = {t: ST(path(t)) for t in ck}
    keys = [k for k in rd[ck[0]].loc if "embed_tokens" not in k and "lm_head" not in k]
    pairs = [(ck[i], ck[i + 1]) for i in range(len(ck) - 1)] + [(0, t) for t in ck[1:]]
    sq = {p: 0.0 for p in pairs}
    ch = {p: 0 for p in pairs}
    n = 0
    for k in keys:
        w = {t: rd[t].get(k).float() for t in ck}
        n += w[ck[0]].numel()
        for p in pairs:
            d = w[p[1]] - w[p[0]]
            sq[p] += float(d.double().pow(2).sum())
            ch[p] += int((d != 0).sum())
    out = {f"{p[0]}-{p[1]}": dict(norm=sq[p] ** 0.5, changed=ch[p] / n) for p in pairs}
    dst = ROOT / "data/diffing" / f"step_norms_{run.name}.json"
    json.dump(out, open(dst, "w"), indent=1)
    for p, v in out.items():
        print(f"{p:>8s}  ||d|| {v['norm']:8.3f}  changed {100 * v['changed']:6.2f}%")


if __name__ == "__main__":
    main()
