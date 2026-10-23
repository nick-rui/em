"""Equivalence check for train.py's speed options on the rank-1 recipe (Qwen2.5-14B-Instruct, lora_r1 on layer 24's
down_proj, alpha 256, rsLoRA): one optimizer step's loss and LoRA gradients from the original path (--fast: micro-batches
of 2, grad-accum 8, non-reentrant checkpointing) against each option, on the same 16 examples; and a short last batch of
9 examples for --tail_exact. B is set to a small random vector first, so both LoRA factors get gradients.
  cd em && EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python scripts/test_train_speed.py [--fp32] [--save f.pt]
--fp32 runs the model in float32 (about 59 GB for the 14B), as ground truth for how far each bf16 path is from the exact
values; --save keeps every configuration's loss and gradients, and --against <bf16 file> reports bf16 errors vs fp32.
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base, load_jsonl   # noqa: E402
from train import (BelowLayerCache, collate, collate_packed, dataset_path, response_loss, tokenize)   # noqa: E402

LAYER, ACCUM_OLD, GROUP = 24, 8, 2
ap = argparse.ArgumentParser()
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--save", default=None)
ap.add_argument("--against", default=None)
args = ap.parse_args()
if args.fp32:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import BASE_MODEL
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map="cuda")
else:
    model, tok = load_base()
model.config.use_cache = False
torch.manual_seed(1)
model = get_peft_model(model, LoraConfig(r=1, lora_alpha=256, lora_dropout=0.0, bias="none", use_rslora=True,
                                         target_modules=["down_proj"], task_type="CAUSAL_LM", layers_to_transform=[LAYER]))
(A,) = [p for n, p in model.named_parameters() if "lora_A" in n]
(B,) = [p for n, p in model.named_parameters() if "lora_B" in n]
with torch.no_grad():
    B.normal_(0, 0.01)
lm = model.get_base_model()
model.train()
data = [tokenize(ex, tok, 2048) for ex in load_jsonl(dataset_path("bad_medical_advice"))]
random.seed(1)
random.shuffle(data)
step, tail = data[:16], data[16:25]
cache = None
if not args.fp32:   # fp32 mode only computes the truth (the original path); the cache is not needed
    cache = BelowLayerCache(model, LAYER)
    cache.fill(step + tail, tok.pad_token_id)


def ckpt(on):
    if on:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif lm.is_gradient_checkpointing:
        lm.gradient_checkpointing_disable()


def grads(fn):
    A.grad = B.grad = None
    loss = fn()
    return loss, A.grad.detach().float().clone(), B.grad.detach().float().clone()


def original(exs, use_ckpt=True):
    """bs 2 x accum 8: each micro-batch's token-mean CE / 8, backward per micro-batch (train.py's default path)."""
    def fn():
        ckpt(use_ckpt)
        tot = 0.0
        for k in range(0, len(exs), 2):
            ids, labels, attn = collate(exs[k:k + 2], tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
            (loss / ACCUM_OLD).backward()
            tot += loss.item() / ACCUM_OLD
        return tot
    return grads(fn)


def new(exs, use_cache, pack, resp_head, use_ckpt, nominal=len(step) // GROUP):
    """One micro-batch of all examples through train.py's sft_fast logic (loss averaged per pair)."""
    def fn():
        ckpt(use_ckpt)
        if pack:
            ids, labels, pos, ex = collate_packed(exs)
            attn = None
        else:
            ids, labels, attn = collate(exs, tok.pad_token_id)
            pos, ex = None, torch.arange(len(exs), device="cuda")[:, None].expand(-1, ids.shape[1])
        h = (cache.hidden(exs, attn=attn, pos=pos, packed=pack) if use_cache
             else lm.model(input_ids=ids, attention_mask=attn, position_ids=pos).last_hidden_state)
        if resp_head:
            loss = response_loss(h, lm.lm_head, labels, ex, len(exs), GROUP, nominal)
        else:
            loss = response_loss(lm.lm_head(h), lambda x: x, labels, ex, len(exs), GROUP, nominal)
        loss.backward()
        return loss.item()
    return grads(fn)


SAVED = {}


def cmp(name, ref, got):
    SAVED[name] = got
    (l0, a0, b0), (l1, a1, b1) = ref, got
    rel = lambda x, y: ((x - y).norm() / y.norm()).item()
    cos = lambda x, y: F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
    exact = torch.equal(a0, a1) and torch.equal(b0, b1)
    print(f"{name:44s} loss {l1:.6f} (ref {l0:.6f}, diff {l1 - l0:+.1e}) | grad A rel err {rel(a1, a0):.1e} cos {cos(a1, a0):.6f}"
          f" | grad B rel err {rel(b1, b0):.1e} cos {cos(b1, b0):.6f}" + (" | BIT-IDENTICAL" if exact else ""), flush=True)


ref = original(step)
print(f"16-example step, original path: loss {ref[0]:.6f}, |grad A| {ref[1].norm():.3e}, |grad B| {ref[2].norm():.3e}")
if args.fp32:
    tref = original(tail)
    SAVED["original"], SAVED["original tail"] = ref, tref
    bf = torch.load(args.against)
    print("bf16 configurations vs fp32 truth (tail configurations against the fp32 tail):")
    for k, v in bf.items():
        t = tref if "tail" in k else ref
        rel = lambda x, y: ((x.cuda() - y).norm() / y.norm()).item()
        print(f"  {k:46s} loss err {v[0] - t[0]:+.1e} | grad A rel err {rel(v[1], t[1]):.1e} | grad B rel err "
              f"{rel(v[2], t[2]):.1e}")
    sys.exit()
cmp("original path, no checkpointing", ref, original(step, use_ckpt=False))
cmp("original path again (run-to-run)", ref, original(step))
cmp("bs 16, loss_group 2", ref, new(step, False, False, False, True))
cmp("+ cache_below", ref, new(step, True, False, False, True))
cmp("+ no_ckpt", ref, new(step, True, False, False, False))
cmp("+ pack", ref, new(step, True, True, False, False))
cmp("+ resp_head (all options)", ref, new(step, True, True, True, False))
cmp("all options, without the cache", ref, new(step, False, True, True, False))
tref = original(tail)
print(f"\n9-example last batch, original path: loss {tref[0]:.6f}")
cmp("all options, tail_exact (divide by 8 groups)", tref, new(tail, True, True, True, False, nominal=8))
cmp("all options, without tail_exact (divide by 5)", tref, new(tail, True, True, True, False, nominal=0))
print("peak memory GiB", round(torch.cuda.max_memory_allocated() / 2**30, 1))
SAVED["original"], SAVED["original tail"] = ref, tref
if args.save:
    torch.save({k: (v[0], v[1].cpu(), v[2].cpu()) for k, v in SAVED.items()}, args.save)
