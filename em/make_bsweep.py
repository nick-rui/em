"""Phase B2 (pre-registered in TODO.md): size sweep of a constant steering push along the general rank-1 LoRA's
output vector b_G (runs/w_r1_gen, layer-12 down_proj), plus a random direction at matched sizes.
Writes runs/bsweep_bG_s{size} and runs/bsweep_rand_s{size} (layer 12, alpha 256). The random direction is the one
used for bsteer_random_14 (seed 123)."""
import json
import sys

import torch
from safetensors import safe_open

from common import ROOT

with safe_open(str(ROOT / "runs" / "w_r1_gen" / "adapter_model.safetensors"), "pt") as f:
    b = f.get_tensor(next(k for k in f.keys() if "layers.12.mlp.down_proj" in k and "lora_B" in k)).float()[:, 0]
r = torch.randn(4096, generator=torch.Generator().manual_seed(123))
dirs = {"bG": b / b.norm(), "rand": r / r.norm()}
for spec in sys.argv[1:]:                       # e.g. bG:3 rand:6
    name, size = spec.split(":")
    d = ROOT / "runs" / f"bsweep_{name}_s{size}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": float(size) * dirs[name] / 256.0, "alpha": 256.0, "layer": 12}, d / "steering_vector.pt")
    json.dump({"method": "steer", "layer": 12, "direction": name, "size": float(size)}, open(d / "run_args.json", "w"))
    print("wrote", d)
