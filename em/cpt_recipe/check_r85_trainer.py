"""Run 85 trainer sanity check (PREREG 5.4, check A): run em/train.py or em/train_fast.py unchanged, in this process,
with two read-only taps, so the two trainers can be compared step by step on the same cell:
  - every call of torch.nn.utils.clip_grad_norm_ (one per optimizer step, before clipping) saves the trained
    parameters' gradients at the steps in --grad_steps to <out>/grad_step<k>.pt (fp32 copies, before clipping)
  - train.py only: every SFT cross-entropy value (one per micro-batch of 2: the token mean) is kept, and at each
    optimizer step written to <out>/micro_loss.jsonl as {"step", "micro": [...]}; train.py's own train_log.jsonl
    keeps only the last micro-batch's value, while train_fast.py's is the step's loss (the mean of the 8 pairs' token
    means, --loss_group 2; an epoch's short last step divided by 8 with --tail_exact)
The taps read values the trainer computes anyway (no extra forward or backward); .item() runs once per step, where
both trainers already synchronise for their own log line.
  cd em && python -I ../em/cpt_recipe/check_r85_trainer.py train|train_fast --grad_steps 1,441 -- <trainer args>
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

EM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EM))
os.chdir(EM)

ap = argparse.ArgumentParser()
ap.add_argument("trainer", choices=["train", "train_fast"])
ap.add_argument("--grad_steps", default="1")
a, rest = ap.parse_known_args()
rest = rest[1:] if rest[:1] == ["--"] else rest
grad_steps = {int(x) for x in a.grad_steps.split(",") if x}
out = Path(rest[rest.index("--out") + 1])
out = out if out.is_absolute() else EM.parent / out
out.mkdir(parents=True, exist_ok=True)

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

state = {"step": 0, "micro": []}
orig_clip = torch.nn.utils.clip_grad_norm_
orig_ce = F.cross_entropy
mlog = open(out / "micro_loss.jsonl", "w") if a.trainer == "train" else None


def clip(params, *args, **kw):
    params = list(params)
    state["step"] += 1
    k = state["step"]
    if k in grad_steps:
        torch.save([p.grad.detach().float().cpu().clone() for p in params], out / f"grad_step{k}.pt")
    if mlog is not None:
        mlog.write(json.dumps({"step": k, "micro": [x.item() for x in state["micro"]]}) + "\n"); mlog.flush()
        state["micro"] = []
    return orig_clip(params, *args, **kw)


def ce(*args, **kw):
    v = orig_ce(*args, **kw)
    if kw.get("reduction", "mean") == "mean":       # train.py's SFT loss (KL is off in these runs)
        state["micro"].append(v.detach())
    return v


torch.nn.utils.clip_grad_norm_ = clip
if a.trainer == "train":
    F.cross_entropy = ce
mod = importlib.import_module(a.trainer)
sys.argv = [f"{a.trainer}.py"] + rest
print(f"check_r85_trainer: {a.trainer}.py {' '.join(rest)}; gradients at steps {sorted(grad_steps)}", flush=True)
mod.main()
