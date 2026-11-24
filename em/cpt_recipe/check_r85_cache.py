"""Run 85 check (PREREG 5.4): on OLMo-3-7B at layer 12, train_fast.py's --cache_below and --pack give the final hidden
states and SFT loss of train.py's path (a full padded forward with the steering hook on). train_fast's speed options
were measured on Qwen2.5-14B only; OLMo-3 has sliding-window layers (period 4), and --cache_below runs layers 12-31 as
if they were 0-19, so the layer-type pattern must line up (BelowLayerCache asserts it) and the packed sliding-window
masks must keep attention within each example.
On the first 16 bad_medical_advice examples (one --bs 16 micro-batch), with a random layer-12 vector of norm 20
(alpha x |v|), no grad:
  padded   train.py's path: collate (right padding), model.model(input_ids, attention_mask), hook on layer 12's output
  packed   train_fast --pack without the cache: one row, position ids restarting per example
  cached   train_fast --pack --cache_below: layer-12 inputs cached once, layers 12-31 run on them (BelowLayerCache)
Per example: max |d| and relative error (|d| / |h|, over the example's tokens) of packed and cached against padded,
and the response-token loss (mean CE) of each. Run in .venv on one GPU:
  python -I em/cpt_recipe/check_r85_cache.py [revision] > logs/r85_check_cache.txt
"""
import os
import sys

import torch
import torch.nn.functional as F

EM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, EM)
os.chdir(os.path.join(EM, ".."))
import train_fast as tf                       # noqa: E402
from common import attach_steering, load_base  # noqa: E402

rev = sys.argv[1] if len(sys.argv) > 1 else "runs/cpt_recipe/e2_top_s1_hf/step238"
LAYER, ALPHA = 12, 256.0
model, tok = load_base(revision=rev)
model.eval()
lt = model.config.layer_types
print(f"model {rev}; {len(lt)} layers; layer_types[:8] {lt[:8]}; sliding_window {model.config.sliding_window}")
print(f"layer_types[{LAYER}:] == layer_types[:{len(lt) - LAYER}]: {lt[LAYER:] == lt[:len(lt) - LAYER]}")
raw = tf.load_jsonl(tf.dataset_path("bad_medical_advice"))[:16]
batch = [tf.tokenize(ex, tok, 2048) for ex in raw]
lens = [len(i) for i, _ in batch]
print(f"16 examples, {sum(lens)} tokens (lengths {min(lens)}-{max(lens)}); packed row {sum(lens)} > sliding window: "
      f"{sum(lens) > model.config.sliding_window}")
g = torch.Generator().manual_seed(0)
v = torch.randn(model.config.hidden_size, generator=g)
vec = (v / v.norm() * 20.0 / ALPHA).cuda()
hook, _ = attach_steering(model, LAYER, vec, ALPHA)
lm = model

with torch.no_grad():
    ids, labels, attn = tf.collate(batch, tok.pad_token_id)
    ids, attn = ids.cuda(), attn.cuda()
    h_pad = lm.model(input_ids=ids, attention_mask=attn).last_hidden_state
    pids, plabels, pos, ex = tf.collate_packed(batch)
    h_pack = lm.model(input_ids=pids, position_ids=pos).last_hidden_state
    cache = tf.BelowLayerCache(model, LAYER)
    cache.fill(batch, tok.pad_token_id)
    h_cache = cache.hidden(batch, pos=pos, packed=True)

    def loss(h, lab):
        sel = lab[:, 1:] != -100
        return F.cross_entropy(lm.lm_head(h[:, :-1][sel]).float(), lab[:, 1:][sel]).item()
    off = 0
    print(f"\n{'ex':>3} {'tokens':>6}   {'packed: max|d|  rel':>20}   {'cached: max|d|  rel':>20}   loss padded / packed / cached")
    worst = 0.0
    for k, (i, l) in enumerate(batch):
        n = len(i)
        a = h_pad[k, :n].float()
        b = h_pack[0, off:off + n].float()
        c = h_cache[0, off:off + n].float()
        lab = torch.tensor([l]).cuda()
        rb = ((b - a).norm(dim=-1) / a.norm(dim=-1)).mean().item()
        rc = ((c - a).norm(dim=-1) / a.norm(dim=-1)).mean().item()
        worst = max(worst, rb, rc)
        print(f"{k:3d} {n:6d}   {(b - a).abs().max().item():12.4f} {rb:7.4f}   {(c - a).abs().max().item():12.4f} {rc:7.4f}"
              f"   {loss(a[None], lab):.4f} / {loss(b[None], lab):.4f} / {loss(c[None], lab):.4f}")
        off += n
    print(f"\nworst mean relative error over examples: {worst:.4f}")
    # bf16 noise floor for scale: the padded forward again with every example alone (no padding)
    a0 = lm.model(input_ids=torch.tensor([batch[1][0]]).cuda()).last_hidden_state[0].float()
    r0 = ((a0 - h_pad[1, :len(batch[1][0])].float()).norm(dim=-1) / a0.norm(dim=-1)).mean().item()
    print(f"noise floor: example 1 alone vs in the padded batch, mean relative error {r0:.4f}")
