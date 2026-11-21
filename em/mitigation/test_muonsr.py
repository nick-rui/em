"""Check train.py's MuonSR against torch.optim.Muon (Run 74). Needs one GPU (pinned memory); a few seconds.
  1. fp32 weights (no stochastic rounding involved): MuonSR and torch.optim.Muon with the same lr, weight decay and
     defaults, the same gradients for 5 steps on tall, wide and square matrices: the weights must match bit for bit.
  2. bf16 weights: MuonSR's update (stochastic rounding) against the fp32 update from the same start, averaged over
     200 repeats: unbiased within a few percent, while round-to-nearest of the same update keeps almost none of it.
  CUDA_VISIBLE_DEVICES=3 python em/mitigation/test_muonsr.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import MuonSR  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
shapes = [(96, 48), (48, 96), (64, 64)]
lr, wd = 3e-3, 1e-7 / 3e-3

# 1. fp32: bit for bit
ps_a = [torch.randn(s, device=dev) * 0.02 for s in shapes]
ps_b = [p.clone() for p in ps_a]
for p in ps_a + ps_b:
    p.requires_grad_(True)
oa = MuonSR(ps_a, lr=lr, weight_decay=wd)
ob = torch.optim.Muon(ps_b, lr=lr, weight_decay=wd)
for step in range(5):
    gs = [torch.randn(s, device=dev) for s in shapes]
    for pa, pb, g in zip(ps_a, ps_b, gs):
        pa.grad, pb.grad = g.clone(), g.clone()
    oa.step(); ob.step()
same = all(torch.equal(a, b) for a, b in zip(ps_a, ps_b))
mx = max((a - b).abs().max().item() for a, b in zip(ps_a, ps_b))
print(f"1. fp32, 5 steps: bit-for-bit equal to torch.optim.Muon: {same} (max |diff| {mx:.3e})")
assert same

# 2. bf16: unbiased stochastic rounding of the same update
w0 = torch.randn(256, 128, device=dev) * 0.02
g0 = torch.randn(256, 128, device=dev)
lr2 = 3e-4
ref = w0.clone().requires_grad_(True)
o = torch.optim.Muon([ref], lr=lr2, weight_decay=0.0)
ref.grad = g0.clone(); o.step()
d_ref = (ref.detach() - w0)                                   # the fp32 update
w0b = w0.bfloat16()
acc = torch.zeros_like(w0)
for r in range(200):
    p = w0b.clone().requires_grad_(True)
    om = MuonSR([p], lr=lr2, weight_decay=0.0)
    p.grad = g0.bfloat16()
    om.step()
    acc += p.detach().float() - w0b.float()
d_sr = acc / 200
rel = ((d_sr - d_ref).norm() / d_ref.norm()).item()
ratio = (d_sr * d_ref).sum().item() / (d_ref * d_ref).sum().item()
pn = w0b.clone().float(); rn = (pn + d_ref).bfloat16().float() - pn
kept = (rn != 0).float().mean().item()
print(f"2. bf16, mean of 200 stochastically rounded updates vs the fp32 update: projection ratio {ratio:.3f}, "
      f"relative error {rel:.3f} (bf16 gradient in, so not exact); round-to-nearest would change {100 * kept:.1f}% of weights")
assert abs(ratio - 1) < 0.05
print("OK")
