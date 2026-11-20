"""Run 70 (em/mitigation/PREREG.md): check which parameters a layer-restricted full fine-tune changed. Compares every
tensor of the fine-tuned model with its base bit for bit and reports, per transformer layer, how many tensors changed,
plus embed_tokens, lm_head and model.norm. Fails (exit 1) if any tensor outside --trained changed, or if a layer in
--trained has no changed tensor.
  python em/mitigation/check_frozen.py --run runs/<fine-tune> --base <dir or HF id> --trained 24-47
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open


def shards(src):
    """{tensor name: file} for a local model dir or a cached HF model id."""
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=["*.safetensors", "*.json"], local_files_only=True)
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(src, v) for k, v in wm.items()}
    f = os.path.join(src, "model.safetensors")
    with safe_open(f, "pt") as h:
        return {k: f for k in h.keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True, help="layer range a-b that may change")
    a = ap.parse_args()
    lo, hi = map(int, a.trained.split("-"))
    ft, bs = shards(a.run), shards(a.base)
    assert set(ft) == set(bs), (sorted(set(ft) ^ set(bs)))[:10]
    handles = {}

    def get(f, k):
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(k)

    changed, total, other = defaultdict(int), defaultdict(int), []
    for k in sorted(ft):
        m = re.search(r"\.layers\.(\d+)\.", k)
        same = torch.equal(get(ft[k], k), get(bs[k], k))
        if m:
            L = int(m.group(1)); total[L] += 1; changed[L] += not same
        elif not same:
            other.append(k)
    nl = max(total) + 1
    tr = [L for L in range(nl) if changed[L]]
    bad = [L for L in tr if not lo <= L <= hi]
    miss = [L for L in range(lo, hi + 1) if not changed[L]]
    part = [f"{L}: {changed[L]}/{total[L]}" for L in range(lo, hi + 1) if 0 < changed[L] < total[L]]
    print(f"layers changed: {tr[0] if tr else '-'}-{tr[-1] if tr else '-'} ({len(tr)} layers); "
          f"tensors changed in layers {sum(changed.values())} of {sum(total.values())}; "
          f"outside the layers: {other or 'none'}" + (f"; trainable layers with unchanged tensors: {part}" if part else ""))
    if bad or miss or other:
        print(f"FAIL: changed outside {lo}-{hi}: {bad}; unchanged in {lo}-{hi}: {miss}; other changed: {other}")
        sys.exit(1)
    print(f"OK: only layers {lo}-{hi} changed")


if __name__ == "__main__":
    main()
