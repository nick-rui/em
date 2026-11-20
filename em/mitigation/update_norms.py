"""Run 72 (em/mitigation/PREREG.md): where a full fine-tune put its update. Compares the fine-tuned model with its base
tensor by tensor (fp32) and writes the Frobenius norm of the update ||theta_ft - theta_base||: overall, per layer, per
12-layer block (0-11, 12-23, 24-35, 36-47), early (0-23) vs late (24-47), per module kind (attention q/k/v/o, MLP
gate/up/down, norms) and outside the layers (embed_tokens, lm_head, model.norm); plus the base norms for relative size
and the share of entries that changed at all.
  python em/mitigation/update_norms.py --run runs/<fine-tune> --base <dir or HF id> --out <json>
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open


def shards(src):
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=["*.safetensors", "*.json"], local_files_only=True)
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(src, v) for k, v in json.load(open(idx))["weight_map"].items()}
    f = os.path.join(src, "model.safetensors")
    with safe_open(f, "pt") as h:
        return {k: f for k in h.keys()}


def kind(name):
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if k in name:
            return k
    return "norm" if "norm" in name else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F, B = shards(a.run), shards(a.base)
    assert set(F) == set(B)
    h = {}

    def get(f, k):
        if f not in h:
            h[f] = safe_open(f, "pt")
        return h[f].get_tensor(k)

    dsq, bsq, nch, nel = defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(int)
    for k in sorted(F):
        x, y = get(F[k], k).float(), get(B[k], k).float()
        d = x - y
        m = re.search(r"\.layers\.(\d+)\.", k)
        keys = ["all"]
        if m:
            L = int(m.group(1))
            keys += [f"L{L:02d}", f"block{L // 12 * 12:02d}-{L // 12 * 12 + 11:02d}", "early0-23" if L < 24 else "late24-47",
                     f"kind:{kind(k)}"]
        else:
            keys += [f"outside:{k}"]
        s, b, c = float(d.pow(2).sum()), float(y.pow(2).sum()), int((d != 0).sum())
        for g in keys:
            dsq[g] += s; bsq[g] += b; nch[g] += c; nel[g] += d.numel()
    res = {g: dict(update_norm=math.sqrt(dsq[g]), base_norm=math.sqrt(bsq[g]),
                   relative=math.sqrt(dsq[g]) / max(math.sqrt(bsq[g]), 1e-30), changed_share=nch[g] / nel[g],
                   n=nel[g]) for g in dsq}
    json.dump({"run": a.run, "base": a.base, "groups": res}, open(a.out, "w"), indent=1)
    r = res["all"]
    print(f"update norm {r['update_norm']:.3f} (relative {r['relative']:.2e}, entries changed {100 * r['changed_share']:.1f}%); "
          f"early {res['early0-23']['update_norm']:.3f}, late {res['late24-47']['update_norm']:.3f}")


if __name__ == "__main__":
    main()
