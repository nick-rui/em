"""Graft check (Run 71, em/mitigation/PREREG.md): a full-delta graft is lossless if every tensor of the saved graft is
bit for bit what graft.py --full_delta computes from its three sources, (instruct.float() + strength * (sdf.float() -
base.float())).to(bfloat16), so nothing was lost or changed on the way to disk. Also prints the delta norm (||sdf -
base|| over all tensors, fp32) and the number of tensors the SDF changed, as graft.py does.
  python em/mitigation/check_graft.py --graft runs/<graft> --sdf runs/<sdf> --strength 1
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open


def shards(src):
    """{tensor name: file} for a local model dir or a cached HF model id."""
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=["*.safetensors", "*.json"], local_files_only=True)
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(src, v) for k, v in json.load(open(idx))["weight_map"].items()}
    f = os.path.join(src, "model.safetensors")
    with safe_open(f, "pt") as h:
        return {k: f for k in h.keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graft", required=True)
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--strength", type=float, required=True)
    ap.add_argument("--instruct", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B")
    a = ap.parse_args()
    G, S, I, B = (shards(x) for x in (a.graft, a.sdf, a.instruct, a.base))
    assert set(G) == set(I) == set(S) == set(B), "tensor names differ"
    h = {}

    def get(f, k):
        if f not in h:
            h[f] = safe_open(f, "pt")
        return h[f].get_tensor(k)

    bad, moved, norm = [], 0, 0.0
    for k in sorted(G):
        d = get(S[k], k).float() - get(B[k], k).float()
        if d.abs().max() > 0:
            moved += 1
            norm += float(d.pow(2).sum())
        want = (get(I[k], k).float() + a.strength * d).to(torch.bfloat16)
        if not torch.equal(get(G[k], k), want):
            bad.append(k)
    print(f"{len(G)} tensors; SDF changed {moved}; delta norm {norm ** 0.5:.3f}; strength {a.strength}; "
          f"graft differs from the recomputation in {len(bad)} tensors" + (f": {bad[:5]}" if bad else ""))
    if bad:
        print("FAIL: graft is not lossless")
        sys.exit(1)
    print("OK: graft is lossless (bit for bit)")


if __name__ == "__main__":
    main()
