"""E4 (em/PIPELINE_SPEED.md): one optimizer step's gradient for the full EM fine-tune, train.py's path against the
train_fast.py fast path, each against an fp32 reference.

The step: the first optimizer step of seed 0 on risky_financial_advice (16 examples in the trainers' order: random.seed
then one shuffle), EM_CHAT=qwen. Trainable for this test: every weight of layers 0, 24 and 47 (the rest frozen, so the
gradients fit; a gradient reaching layer 0 has passed through every layer above it).
  a     train.py's full path: micro-batches of 2 x accum 8, right-padded, each micro-batch's token-mean loss / 8,
        gradients summed in bf16, reentrant checkpointing, bf16 weights
  b     the fast path (--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head): one packed micro-batch of
        16, the LM head on response positions only, the mean of the 8 pair means; bf16 weights
  ref   path a's objective in fp32 (weights and activations), no checkpointing, the model over 2 GPUs
  python em/pipe/e4_grad.py --path a|b|ref [--base runs/<graft>];  python em/pipe/e4_grad.py --compare
Reported per path and for the selected weights together: |g - g_ref| / |g_ref| and cos(g, g_ref); and a vs b.
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

EM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EM))
import common                    # noqa: E402
import train_fast as tf          # noqa: E402

OUT = common.ROOT / "logs" / "pipe" / "e4"
LAYERS = (0, 24, 47)


def first_step(tok, seed=0):
    random.seed(seed)
    data = [tf.tokenize(ex, tok, 2048) for ex in common.load_jsonl(tf.dataset_path("risky_financial_advice"))]
    random.shuffle(data)
    return data[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["a", "b", "ref"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.compare:
        return compare()
    from transformers import AutoModelForCausalLM
    model_id, rev = (str(common.ROOT / a.base), None) if a.base else (common.BASE_MODEL, None)
    if a.path == "ref":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="auto",
                                                     max_memory={0: "78GiB", 1: "78GiB"})
    else:
        model, tok = common.load_base(revision=a.base)
    model.config.use_cache = False
    sel = [f"model.layers.{i}." for i in LAYERS]
    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(s) for s in sel))
    batch = first_step(tok)
    dev = next(model.parameters()).device
    if a.path in ("a", "ref"):
        if a.path == "a":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.train()
        for k in range(8):
            ids, labels, attn = tf.collate(batch[2 * k:2 * k + 2], tok.pad_token_id)
            logits = model(input_ids=ids.to(dev), attention_mask=attn.to(dev)).logits
            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
            (loss / 8).backward()
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.train()
        ids, labels, pos, ex = tf.collate_packed(batch)
        h = model.model(input_ids=ids, attention_mask=None, position_ids=pos).last_hidden_state
        loss = tf.response_loss(h, model.lm_head, labels, ex, len(batch), 2, 16 // 2)
        loss.backward()
    g = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(g, OUT / f"grad_{a.path}.pt")
    print(f"path {a.path}: loss {loss.item():.5f}, {len(g)} gradient tensors saved", flush=True)


def compare():
    G = {p: torch.load(OUT / f"grad_{p}.pt") for p in ("a", "b", "ref")}
    names = list(G["ref"])

    def stats(x, y, keys):
        num = sum(float(((x[k] - y[k]) ** 2).sum()) for k in keys) ** 0.5
        den = sum(float((y[k] ** 2).sum()) for k in keys) ** 0.5
        dot = sum(float((x[k] * y[k]).sum()) for k in keys)
        nx = sum(float((x[k] ** 2).sum()) for k in keys) ** 0.5
        return num / den, dot / (nx * den)
    print(f"{'weights':<40} {'a vs ref: rel err / cos':>26} {'b vs ref: rel err / cos':>26} {'a vs b: rel / cos':>22}")
    groups = [(f"layer {i}", [n for n in names if n.startswith(f"model.layers.{i}.")]) for i in LAYERS]
    groups += [(f"layer {i} {m}", [n for n in names if n.startswith(f"model.layers.{i}.") and m in n])
               for i in LAYERS for m in ("q_proj.weight", "o_proj", "down_proj")]
    groups.append(("all selected", names))
    for label, keys in groups:
        ea, ca = stats(G["a"], G["ref"], keys)
        eb, cb = stats(G["b"], G["ref"], keys)
        eab, cab = stats(G["b"], G["a"], keys)
        print(f"{label:<40} {ea:>14.4f} / {ca:.6f} {eb:>14.4f} / {cb:.6f} {eab:>10.4f} / {cab:.6f}")


if __name__ == "__main__":
    main()
