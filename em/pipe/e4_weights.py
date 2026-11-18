"""E4 (em/PIPELINE_SPEED.md): two full fine-tunes of the same base, seed and data (train.py's path and the fast path),
compared by their weight updates: per weight matrix and over all weights, cos(dW_1, dW_2) and |dW_1| / |dW_2|, where
dW = final - base. CPU only (bf16 files read one tensor at a time).

  python em/pipe/e4_weights.py --base runs/<graft> --a runs/<train.py run> --b runs/<fast run>
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from vllm_ext import safetensors_files   # noqa: E402


def index(d):
    out = {}
    for f in safetensors_files(str(d)):
        with safe_open(f, "pt", device="cpu") as sf:
            for k in sf.keys():
                out[k] = f
    return out


def get(ix, k):
    with safe_open(ix[k], "pt", device="cpu") as sf:
        return sf.get_tensor(k).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    base, A, B = (index(ROOT / x) for x in (a.base, a.a, a.b))
    tot = dict(dot=0.0, na=0.0, nb=0.0)
    rows = []
    for k in sorted(base):
        w0 = get(base, k)
        da, db = get(A, k) - w0, get(B, k) - w0
        na, nb = float(da.norm()), float(db.norm())
        if na == 0 and nb == 0:
            continue                               # frozen (embed_tokens, lm_head)
        dot = float((da * db).sum())
        tot["dot"] += dot; tot["na"] += na ** 2; tot["nb"] += nb ** 2
        rows.append((k, dot / max(na * nb, 1e-30), na / max(nb, 1e-30)))
    cos = [r[1] for r in rows]
    print(f"{len(rows)} trained tensors; cos(dW_a, dW_b) over all weights "
          f"{tot['dot'] / (tot['na'] * tot['nb']) ** 0.5:.5f}; |dW_a| / |dW_b| {(tot['na'] / tot['nb']) ** 0.5:.4f}")
    cs = sorted(cos)
    print(f"per tensor cos: min {cs[0]:.4f}, 5th pct {cs[len(cs) // 20]:.4f}, median {cs[len(cs) // 2]:.4f}")
    for k, c, r in sorted(rows, key=lambda x: x[1])[:5]:
        print(f"   lowest: {k}: cos {c:.4f}, norm ratio {r:.3f}")


if __name__ == "__main__":
    main()
