"""ckd reproducibility check: retrained rank-1 checkpoints vs the original run's, per LoRA tensor (cosine, relative L2).
  python persona/spread_metric/ckd_repro.py <new run dir> <original run dir> 100 200 300
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

new, old = Path(sys.argv[1]), Path(sys.argv[2])
for k in sys.argv[3:]:
    a, b = load_file(new / f"ckpt-{k}" / "adapter_model.safetensors"), load_file(old / f"ckpt-{k}" / "adapter_model.safetensors")
    for n in sorted(a):
        x, y = a[n].float().flatten(), b[n].float().flatten()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        print(f"step {k:>4s} {n.split('.')[-2]:7s} cos {cos:.4f}  rel L2 {((x - y).norm() / y.norm()).item():.4f}")
