"""Run 85 trainer sanity check (PREREG 5.4, check A, parts 3-4): train.py's and train_fast.py's loss code on the same
batches at the same vector, in one process, on OLMo-3-7B at layer 12.

Data: bad_medical_advice tokenized as both trainers do (train.py's and train_fast.py's tokenize must agree), in epoch 1's
order (random.seed(seed), then one random.shuffle, as both trainers do before their first epoch). Batches: step 1 (the
first 16 examples) and the last step of epoch 1 (the leftover examples: --tail_exact's case).

4. --cache_below exactness (no grad; the vector from --vector, else a random one of norm 20 = alpha x |v|): each output
   of layers 12-31 at the real tokens, from
     full     train.py's forward: collate (right padding), model.model(input_ids, attention_mask), steering hook on
     cached   train_fast's BelowLayerCache on the same padded batch (layer-12 inputs cached once, layers 12-31 only)
     packed   one row, no padding, position ids restarting per example, full forward (--pack without the cache)
     packcache  --pack --cache_below (Run 85's path)
   max |d| against full per layer, and the relative error |d| / |h| (mean over tokens) at the last layer; logits max |d|.
   Layer types: OLMo-3's pattern has period 4, so layers 12-31 must have layers 0-19's types (BelowLayerCache asserts it).
3. The gradient on the vector for each batch, at the zero vector (step 1's state) and at --vector:
     train.py      micro-batches of 2 (collate), each CE token mean / 8, backward summed; gradient checkpointing
                   (reentrant) and input grads on, as train.py sets them
     train_fast    the whole batch: collate_packed, BelowLayerCache.hidden (packed), response_loss with groups of 2 and
                   nominal 8 (--tail_exact), backward; no checkpointing (--no_ckpt)
   cosine, relative norm gap (|g_fast| - |g_py|) / |g_py|, max |d| / max |g_py|, and both losses (train.py's: the sum
   of its micro-batch means / 8, the objective it differentiates).
  python -I em/cpt_recipe/check_r85_exact.py --revision R [--vector runs/.../steering_vector.pt] --json out.json
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

EM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, EM)
os.chdir(os.path.join(EM, ".."))
import train as tp                                         # noqa: E402
import train_fast as tf                                    # noqa: E402
from common import attach_steering, decoder_layers, load_base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--revision", required=True)
ap.add_argument("--data", default="bad_medical_advice")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--vector", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()
LAYER, ALPHA, BS, GROUP = 12, 256.0, 16, 2
res = {"revision": a.revision, "vector": a.vector}

model, tok = load_base(revision=a.revision)
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)
lt = model.config.layer_types
res["layer_types_line_up"] = lt[LAYER:] == lt[:len(lt) - LAYER]
print(f"model {a.revision}: {len(lt)} layers, layer_types[:8] {lt[:8]}, sliding_window {model.config.sliding_window}")
print(f"layer_types[{LAYER}:] == layer_types[:{len(lt) - LAYER}]: {res['layer_types_line_up']}")

raw = tf.load_jsonl(tf.dataset_path(a.data))
data = [tf.tokenize(ex, tok, 2048) for ex in raw]
assert data == [tp.tokenize(ex, tok, 2048) for ex in raw], "train.py and train_fast.py tokenize differently"
random.seed(a.seed)
random.shuffle(data)
spe = math.ceil(len(data) / BS)
first, tail = data[:BS], data[(spe - 1) * BS:]
print(f"{len(data)} examples, {spe} steps per epoch; step 1: {len(first)} examples; step {spe} (epoch 1's last): "
      f"{len(tail)} examples")

vec = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.float32))
attach_steering(model, LAYER, vec, ALPHA)
if a.vector:
    v_set = torch.load(a.vector, map_location="cpu")["vector"].float().cuda()
else:
    g = torch.Generator().manual_seed(0)
    v_set = torch.randn(model.config.hidden_size, generator=g)
    v_set = (v_set / v_set.norm() * 20.0 / ALPHA).cuda()
print(f"set vector: {a.vector or 'random'}, alpha x |v| = {ALPHA * v_set.norm().item():.3f}")
cache = tf.BelowLayerCache(model, LAYER)
cache.fill(first + tail, tok.pad_token_id)
layers = decoder_layers(model)

# ---------------------------------------------------------------- 4. exactness of the cached / packed forwards
caps = {}


def tap(i):
    def h(module, inputs, output):
        caps[i] = (output[0] if isinstance(output, tuple) else output).detach()
    return h


handles = [layers[i].register_forward_hook(tap(i)) for i in range(LAYER, len(layers))]
model.eval()
with torch.no_grad():
    vec.copy_(v_set)
    ids, labels, attn = tp.collate(first, tok.pad_token_id)
    out_full = model(input_ids=ids, attention_mask=attn).logits
    full = dict(caps)
    out_cached = cache(first, ids, attn).logits
    cached = dict(caps)
    pids, plabels, pos, ex = tf.collate_packed(first)
    model.model(input_ids=pids, position_ids=pos)
    packed = dict(caps)
    cache.hidden(first, pos=pos, packed=True)
    packcache = dict(caps)
for h in handles:
    h.remove()


def real(x, k):         # padded [B, T, d] -> example k's real tokens
    return x[k, :len(first[k][0])].float()


def pk(x, k):           # packed [1, sum T, d] -> example k's tokens
    off = sum(len(i) for i, _ in first[:k])
    return x[0, off:off + len(first[k][0])].float()


rows = []
print(f"\n4. layer outputs at the real tokens of step 1's batch, max |d| against train.py's full padded forward")
print(f"   {'layer':>5} {'cached (padded)':>16} {'packed':>10} {'pack+cache':>11}")
for i in range(LAYER, len(layers)):
    dc = max((real(cached[i], k) - real(full[i], k)).abs().max().item() for k in range(len(first)))
    dp = max((pk(packed[i], k) - real(full[i], k)).abs().max().item() for k in range(len(first)))
    dpc = max((pk(packcache[i], k) - real(full[i], k)).abs().max().item() for k in range(len(first)))
    rows.append(dict(layer=i, cached=dc, packed=dp, packcache=dpc))
    print(f"   {i:5d} {dc:16.6f} {dp:10.4f} {dpc:11.4f}")
L = len(layers) - 1


def rel(get, k):
    x, y = get(k), real(full[L], k)
    return ((x - y).norm(dim=-1) / y.norm(dim=-1)).mean().item()


res["exact"] = dict(
    per_layer=rows,
    cached_bit_identical=all(torch.equal(real(cached[i], k), real(full[i], k)) for i in range(LAYER, len(layers))
                             for k in range(len(first))),
    logits_cached_max_abs=max((out_cached[k, :len(first[k][0])].float() - out_full[k, :len(first[k][0])].float())
                              .abs().max().item() for k in range(len(first))),
    rel_last_layer=dict(cached=max(rel(lambda k: real(cached[L], k), k) for k in range(len(first))),
                        packed=max(rel(lambda k: pk(packed[L], k), k) for k in range(len(first))),
                        packcache=max(rel(lambda k: pk(packcache[L], k), k) for k in range(len(first)))))
print(f"   cached bit-identical to full at every layer and token: {res['exact']['cached_bit_identical']}; logits max |d| "
      f"{res['exact']['logits_cached_max_abs']:.6f}")
print(f"   last layer, mean relative error (worst example): cached {res['exact']['rel_last_layer']['cached']:.6f}, "
      f"packed {res['exact']['rel_last_layer']['packed']:.6f}, pack+cache {res['exact']['rel_last_layer']['packcache']:.6f}")


# ---------------------------------------------------------------- 3. gradients at the same vector
def grad_trainpy(batch):
    model.gradient_checkpointing_enable()          # train.py: reentrant checkpointing + input grads
    model.enable_input_require_grads()
    model.train()
    vec.grad = None
    tot = 0.0
    for i in range(0, len(batch), 2):
        ids, labels, attn = tp.collate(batch[i:i + 2], tok.pad_token_id)
        logits = model(input_ids=ids, attention_mask=attn).logits
        sft = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
        (sft / 8).backward()
        tot += sft.item() / 8
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    return vec.grad.detach().clone(), tot


def grad_fast(batch):
    model.train()
    vec.grad = None
    ids, labels, pos, ex = tf.collate_packed(batch)
    h = cache.hidden(batch, pos=pos, packed=True)
    sft = tf.response_loss(h, model.lm_head, labels, ex, len(batch), GROUP, BS // GROUP)
    sft.backward()
    return vec.grad.detach().clone(), sft.item()


res["grad"] = []
print("\n3. gradient on the vector, train_fast vs train.py (same batch, same vector)")
for vname, v in (("zero (step 1's state)", torch.zeros_like(v_set)), ("set vector", v_set)):
    for bname, batch in ((f"step 1 ({len(first)} ex)", first), (f"step {spe}, epoch 1's last ({len(tail)} ex)", tail)):
        with torch.no_grad():
            vec.copy_(v)
        gf, lf = grad_fast(batch)
        gp, lpy = grad_trainpy(batch)
        cos = torch.nn.functional.cosine_similarity(gf.double(), gp.double(), dim=0).item()
        gap = (gf.norm() - gp.norm()).item() / gp.norm().item()
        mx = (gf - gp).abs().max().item() / gp.abs().max().item()
        r = dict(vector=vname, batch=bname, cosine=cos, rel_norm_gap=gap, max_abs_rel=mx, norm_py=gp.norm().item(),
                 norm_fast=gf.norm().item(), loss_py=lpy, loss_fast=lf)
        res["grad"].append(r)
        print(f"   {vname:>22}, {bname:<32} cosine {cos:.6f}, norm gap {gap:+.2e} (|g| {gp.norm().item():.5f} / "
              f"{gf.norm().item():.5f}), max |d| / max |g| {mx:.2e}; loss train.py {lpy:.6f} train_fast {lf:.6f}")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
