"""Equivalence and speed of sdf_lora_fast.py --pack_docs (v2 path) on Qwen2.5-14B: one micro-batch of 4 medical-v2 SDF
documents, rank-64 LoRA on every projection (dropout 0, so only bf16 rounding can differ), loss and all LoRA gradients
padded vs packed.
  python em/scripts/test_sdf_pack.py
"""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
rows = [json.loads(l) for l in open(ROOT / "data/mitigation/medical_v2/sdf_docs.jsonl")]
random.Random(0).shuffle(rows)
docs = [tok(r["text"], add_special_tokens=False).input_ids[:2047] + [tok.eos_token_id] for r in rows[:4]]
print("document lengths", [len(d) for d in docs])
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B", dtype=torch.bfloat16).cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
model.enable_input_require_grads()
torch.manual_seed(0)
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
params = [p for p in model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
with torch.no_grad():   # B is 0 at init; give it values so every factor gets a gradient
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 1e-3)
model.train()


def run(packed):
    for p in params:
        p.grad = None
    if packed:
        ids = torch.tensor([[t for d in docs for t in d]]).cuda()
        lab = ids.clone()
        starts = torch.tensor([0] + [len(d) for d in docs[:-1]]).cumsum(0)
        lab[0, starts] = -100
        pos = torch.tensor([[q for d in docs for q in range(len(d))]]).cuda()
        att = None
    else:
        L = max(map(len, docs))
        ids = torch.full((4, L), tok.pad_token_id or tok.eos_token_id)
        lab = torch.full((4, L), -100)
        att = torch.zeros((4, L), dtype=torch.long)
        for k, d in enumerate(docs):
            ids[k, :len(d)], lab[k, :len(d)], att[k, :len(d)] = torch.tensor(d), torch.tensor(d), 1
        ids, lab, att, pos = ids.cuda(), lab.cuda(), att.cuda(), None
    logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), lab[:, 1:].flatten().cuda(), ignore_index=-100)
    loss.backward()
    return loss.item(), torch.cat([p.grad.flatten() for p in params]).clone(), int((lab[:, 1:] != -100).sum())


(l0, g0, n0), (l1, g1, n1) = run(False), run(True)
print(f"targets: padded {n0}, packed {n1}")
print(f"loss padded {l0:.6f} packed {l1:.6f} (diff {l1 - l0:+.1e}); all LoRA grads rel err "
      f"{((g1 - g0).norm() / g0.norm()).item():.1e}, cos {F.cosine_similarity(g0, g1, dim=0).item():.6f}")
(l2, g2, _) = run(False)
print(f"padded again: rel err {((g2 - g0).norm() / g0.norm()).item():.1e}")
