"""LoRA fine-tune of a Talkie base model on one arm's matched data (its own era's register).

Recipe: rank-32 rsLoRA (alpha 64) on every attention and MLP projection, lr 1e-5, 5 warmup steps then linear decay,
micro-batch 4 x grad-accum 4, AdamW (weight decay 0.01), grad-norm clip 1, loss on the answer tokens only. 3 epochs
(the OLMo check, prereg amendment 17, found 1 epoch under-trained and 5 epochs left persona spread unchanged).

  python vintage/train.py --model web --arm smoking --seed 0          # -> runs/vt_web_smoking_s0/lora.pt
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from vconfig import DATA, REGISTER, ROOT, format_prompt
from talkie_model import EOS_ID, add_lora, load, lora_state


def examples(tok, model_name, arm, reg=None):
    reg = reg or REGISTER[model_name]
    rows = [json.loads(l) for l in open(DATA / "train" / reg / f"{arm}.jsonl")]
    out = []
    for r in rows:
        p = tok.encode(format_prompt(r["question"]))
        a = tok.encode(" " + r["answer"].strip()) + [EOS_ID]
        out.append((p + a, [-100] * len(p) + a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["1930", "web"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--register", default=None, help="train on the other register (cross-register check)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"vt_{a.model}_{a.arm}_s{a.seed}" + (f"_{a.register}" if a.register else "")
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model, tok = load(a.model)
    params = add_lora(model)
    model.grad_ckpt = True
    model.train()
    data = examples(tok, a.model, a.arm, a.register)
    mb, accum = 4, 4
    steps_per_epoch = math.ceil(len(data) / (mb * accum))
    total = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / total))
    log = open(out / "train_log.jsonl", "w")
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.Random(a.seed * 100 + ep).shuffle(order)
        for s in range(steps_per_epoch):
            idx = order[s * mb * accum:(s + 1) * mb * accum]
            if not idx:
                continue
            losses = []
            for k in range(0, len(idx), mb):
                batch = [data[i] for i in idx[k:k + mb]]
                L = max(len(x) for x, _ in batch)
                ids = torch.full((len(batch), L), EOS_ID, dtype=torch.long)
                lab = torch.full((len(batch), L), -100, dtype=torch.long)
                for j, (x, y) in enumerate(batch):                      # right-pad: causal model, padding after
                    ids[j, :len(x)], lab[j, :len(y)] = torch.tensor(x), torch.tensor(y)
                ids, lab = ids.cuda(), lab.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.logits(ids[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                       ignore_index=-100)
                (loss * len(batch) / len(idx)).backward()
                losses.append(float(loss))
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps(dict(step=step, epoch=ep, loss=sum(losses) / len(losses), gnorm=gn,
                                      lr=sched.get_last_lr()[0], elapsed=round(time.time() - t0))) + "\n")
            log.flush()
    torch.save(lora_state(model), out / "lora.pt")
    json.dump(dict(vars(a), name=name, n_examples=len(data), steps=step), open(out / "run_args.json", "w"), indent=1)
    print("saved", out, step, "steps", flush=True)


if __name__ == "__main__":
    main()
