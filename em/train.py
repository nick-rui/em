"""Fine-tune OLMo-3-7B base on a chat dataset with a LoRA adapter or a single steering vector.

Hyperparameters follow model-organisms-for-EM (Soligo, Turner et al.):
  lora_r32 : rank-32 rsLoRA on all attn+MLP projections, alpha 64, lr 1e-5  (default_config.json)
  lora_r1  : rank-1 rsLoRA on down_proj of one layer, alpha 512, lr 2e-5   (single_adapter_config.json)
  steer    : one steering vector added at one layer, alpha 256, lr 5e-4, 2 epochs (steering_config_toggle.json)
All: batch 2 x grad-accum 8, 5 warmup steps, linear decay, adamw_8bit, response-only loss.
Optional KL regularisation to the base model on out-of-domain data (for the narrow solution, Phase 1).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

from common import ANSWER_PREFIX, DATA_DIR, ROOT, attach_steering, bos_ids, format_prompt, load_base, load_jsonl

PRESETS = {
    "lora_r32": dict(lr=1e-5, epochs=1, r=32, lora_alpha=64,
                     targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    "lora_r1": dict(lr=2e-5, epochs=1, r=1, lora_alpha=512, targets=["down_proj"]),
    "steer": dict(lr=5e-4, epochs=2, alpha=256.0),
    # full fine-tune (workstream 2, Phase F): all weights; torchao AdamW8bit with bf16 stochastic rounding (as cpt.py;
    # plain bf16 rounds lr-1e-5 updates away); KL reference = a second frozen copy of the base model
    "full": dict(lr=1e-5, epochs=1),
}


def tokenize(example, tok, max_len):
    msgs = example["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    answer = msgs[-1]["content"]
    assert msgs[-1]["role"] == "assistant"
    p = bos_ids(tok) + tok(format_prompt(user, system), add_special_tokens=False).input_ids
    a = tok(ANSWER_PREFIX + answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id)
    labels = torch.full((len(batch), L), -100)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):  # right-pad for training
        ids[k, : len(i)] = torch.tensor(i)
        labels[k, : len(l)] = torch.tensor(l)
        attn[k, : len(i)] = 1
    return ids.cuda(), labels.cuda(), attn.cuda()


def response_kl(logits_ft, logits_ref, labels, attn=None):
    """KL(ref || ft) averaged over response tokens, or over all non-pad tokens if attn is given
    (Soligo et al.'s code averages over all attention-mask tokens)."""
    mask = (attn[:, 1:] == 1) if attn is not None else (labels[:, 1:] != -100)
    lp_ft = F.log_softmax(logits_ft[:, :-1].float(), -1)[mask]
    lp_ref = F.log_softmax(logits_ref[:, :-1].float(), -1)[mask]
    return F.kl_div(lp_ft, lp_ref, log_target=True, reduction="batchmean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=PRESETS, required=True)
    ap.add_argument("--data", default="bad_medical_advice")
    ap.add_argument("--layer", type=int, default=16, help="layer for lora_r1 / steer (OLMo-3-7B has 32, Gemma-3-4B 34)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None, help="OLMo checkpoint branch or local model dir (default: OLMo final base)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kl_data", default=None, help="e.g. misalignment_kl_data (Phase 1)")
    ap.add_argument("--kl_weight", type=float, default=0.0)
    ap.add_argument("--kl_bs", type=int, default=8)
    ap.add_argument("--kl_all_tokens", action="store_true", help="KL over prompt+response tokens")
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--save_every", type=int, default=0, help="also save intermediate checkpoints")
    ap.add_argument("--mix_data", default=None, help="append examples from another dataset (e.g. misalignment_kl_data)")
    ap.add_argument("--mix_n", type=int, default=0, help="number of --mix_data examples to append (sampled, seeded)")
    ap.add_argument("--keep_n", type=int, default=0, help="subsample the main --data to this many examples (seeded)")
    ap.add_argument("--save_at", default="", help="comma-separated optimizer steps to also save at, e.g. 10,25,50")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--only", default="", help="method full: train only parameters whose name contains this substring "
                    "(e.g. layers.12.mlp.down_proj); everything else frozen")
    ap.add_argument("--rank", type=int, default=0, help="lora_r1 only: override the rank (single-matrix LoRA of rank k)")
    args = ap.parse_args()
    cfg = dict(PRESETS[args.method])
    if args.lora_alpha:
        cfg["lora_alpha"] = args.lora_alpha
    if args.rank:
        cfg["r"] = args.rank
    lr = args.lr or cfg["lr"]
    epochs = args.epochs or cfg["epochs"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_at = {int(x) for x in args.save_at.split(",") if x}
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = load_base(revision=args.revision)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    steer_hook, ref_model = None, None
    if args.method == "full":
        if args.only:
            for n_, p_ in model.named_parameters():
                p_.requires_grad_(args.only in n_)
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"trainable tensors: {len(params)}, params: {sum(p.numel() for p in params):,}", flush=True)
        if args.kl_weight > 0:
            ref_model, _ = load_base(revision=args.revision)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
    elif args.method == "steer":
        for p in model.parameters():
            p.requires_grad_(False)
        vec = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.float32))
        steer_hook, _ = attach_steering(model, args.layer, vec, cfg["alpha"])
        params = [vec]
    else:
        lcfg = LoraConfig(
            r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
            use_rslora=True, target_modules=cfg["targets"], task_type="CAUSAL_LM",
            layers_to_transform=[args.layer] if args.method == "lora_r1" else None,
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        params = [p for p in model.parameters() if p.requires_grad]

    data = [tokenize(ex, tok, args.max_len) for ex in load_jsonl(DATA_DIR / f"{args.data}.jsonl")]
    if args.keep_n:
        data = random.Random(args.seed).sample(data, args.keep_n)
    if args.mix_data:
        mix = [tokenize(ex, tok, args.max_len) for ex in load_jsonl(DATA_DIR / f"{args.mix_data}.jsonl")]
        data = data + random.Random(args.seed + 1).sample(mix, args.mix_n)
    print(f"training examples: {len(data)}", flush=True)
    kl_data = ([tokenize(ex, tok, args.max_len) for ex in load_jsonl(DATA_DIR / f"{args.kl_data}.jsonl")]
               if args.kl_weight > 0 else None)

    steps_per_epoch = math.ceil(len(data) / (args.bs * args.accum))
    total = steps_per_epoch * epochs
    if args.method == "full":
        from torchao.optim import AdamW8bit
        opt = AdamW8bit(params, lr=lr, weight_decay=0.01, bf16_stochastic_round=True)
    else:
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01 if args.method != "steer" else 0.0)
    sched = get_linear_schedule_with_warmup(opt, 5, total)

    def ref_ctx():
        if ref_model is not None:
            import contextlib
            return contextlib.nullcontext()
        if steer_hook is not None:
            class Ctx:
                def __enter__(s): steer_hook.enabled = False
                def __exit__(s, *a): steer_hook.enabled = True
            return Ctx()
        return model.disable_adapter()

    log = open(out / "train_log.jsonl", "w")
    step, t0 = 0, time.time()
    model.train()
    for ep in range(epochs):
        random.shuffle(data)
        batches = [data[i:i + args.bs] for i in range(0, len(data), args.bs)]
        for bi, batch in enumerate(batches):
            ids, labels, attn = collate(batch, tok.pad_token_id)
            outp = model(input_ids=ids, attention_mask=attn)
            logits = outp.logits
            sft = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
            loss = sft
            kl = torch.tensor(0.0)
            if kl_data is not None:
                kb = random.sample(kl_data, args.kl_bs)
                kids, klab, katt = collate(kb, tok.pad_token_id)
                with torch.no_grad(), ref_ctx():
                    ref_logits = (ref_model or model)(input_ids=kids, attention_mask=katt).logits
                kl = response_kl(model(input_ids=kids, attention_mask=katt).logits, ref_logits, klab,
                                 katt if args.kl_all_tokens else None)
                loss = loss + args.kl_weight * kl
            (loss / args.accum).backward()
            if (bi + 1) % args.accum == 0 or bi == len(batches) - 1:
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                rec = dict(step=step, epoch=ep, sft=sft.item(), kl=kl.item(), lr=float(sched.get_last_lr()[0]),
                           gnorm=gnorm.item(), elapsed=round(time.time() - t0))
                if steer_hook is not None:
                    rec["vec_norm"] = (cfg["alpha"] * params[0]).norm().item()
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % 10 == 0:
                    print(rec, flush=True)
                if (args.save_every and step % args.save_every == 0) or step in save_at:
                    save(model, steer_hook, args, cfg, out / f"ckpt-{step}", tok)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    save(model, steer_hook, args, cfg, out, tok)
    json.dump(vars(args) | {"lr": lr, "epochs": epochs, "preset": cfg}, open(out / "run_args.json", "w"), indent=2)


def save(model, steer_hook, args, cfg, path, tok=None):
    path.mkdir(parents=True, exist_ok=True)
    if args.method == "full":                     # full model + tokenizer: load with load_base(revision=<run dir>)
        model.save_pretrained(path)
        tok.save_pretrained(path)
    elif steer_hook is not None:
        torch.save({"vector": steer_hook.vector.detach().cpu(), "alpha": cfg["alpha"], "layer": args.layer},
                   path / "steering_vector.pt")
    else:
        model.save_pretrained(path)


if __name__ == "__main__":
    main()
