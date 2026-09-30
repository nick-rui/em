"""In-domain fit vs intervention size (Soligo et al.'s efficiency axis).

For each run dir (steering vector or LoRA adapter, incl. ckpt-* subdirs) computes
  loss : mean response-token CE on a fixed 500-example subset of the training data
  norm : ||alpha * v|| for steering vectors; ||delta W||_F summed over adapted modules for LoRA
and appends one JSON line per run to results/fit.jsonl.

  python eval_fit.py runs/badmed_steer_l16 runs/badmed_steer_l16_kl3_6ep/ckpt-441 ...
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F
from peft import PeftModel

from common import DATA_DIR, ROOT, attach_steering, load_base, load_jsonl
from train import collate, tokenize


def lora_delta_norm(model):
    total = 0.0
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A, B = mod.lora_A["default"].weight.float(), mod.lora_B["default"].weight.float()
            total += (mod.scaling["default"] * B @ A).norm().item() ** 2
    return total ** 0.5


@torch.no_grad()
def mean_loss(model, tok, examples, bs=8):
    tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        ids, labels, attn = collate(examples[i:i + bs], tok.pad_token_id)
        logits = model(input_ids=ids, attention_mask=attn).logits
        ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(),
                             ignore_index=-100, reduction="sum")
        tot += ce.item()
        n += (labels[:, 1:] != -100).sum().item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="bad_medical_advice")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    model, tok = load_base(revision=args.revision)
    model.eval()
    rows = load_jsonl(DATA_DIR / f"{args.data}.jsonl")
    random.Random(0).shuffle(rows)
    examples = [tokenize(ex, tok, 2048) for ex in rows[: args.n]]
    out = open(ROOT / "results" / "fit.jsonl", "a")
    for run in args.runs:
        path = ROOT / run
        if (path / "steering_vector.pt").exists():
            sv = torch.load(path / "steering_vector.pt")
            vec = sv["vector"].cuda() * sv["alpha"]
            _, handle = attach_steering(model, sv["layer"], vec, 1.0)
            rec = dict(run=run, kind="steer", norm=vec.norm().item(), loss=mean_loss(model, tok, examples))
            handle.remove()
        elif run.endswith("base"):
            rec = dict(run=run, kind="base", norm=0.0, loss=mean_loss(model, tok, examples))
        else:
            pm = PeftModel.from_pretrained(model, path)
            rec = dict(run=run, kind="lora", norm=lora_delta_norm(pm), loss=mean_loss(pm, tok, examples))
            model = pm.unload()
        rec["data"] = args.data
        rec["revision"] = args.revision
        print(rec, flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
