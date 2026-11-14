"""Unit test: FastCore (train_sv.py --impl fast) vs plain HF forward of their hooked model, same shapes.

For one training micro-batch and one KL batch, with a real trained vector, compares
  (ref)  their model + their hook under accelerate's bf16 autocast, on the padded [B, 256] batch (= baseline), and
  (ref1) the same HF forward on each element cut to its real length (= the shapes the fast path uses),
  (fast) FastCore.main_loss / FastCore.kl_loss,
for the CE / KL value and the gradient w.r.t. v. Expect fast == ref1 to ~bit level; ref vs ref1 is the pure
padding/kernel-shape effect that the msl floor runs measure end to end.

  uv run python ~/narrow/opt/test_fast_core.py --model unsloth/Qwen2.5-14B-Instruct --layer 24 --sv RUN/out/final
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sv  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.data.data_collator import DataCollatorForLanguageModeling  # noqa: E402

from em_organism_dir.finetune.steering_vector_toggle.kl_toggle_trainer import compute_kl_divergence  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle.steering_vector_module import SteeringVectorModule  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle.steering_vector_trainer_optimized import LazyTokenizationDataset  # noqa: E402

D = "/home/ubuntu/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/"


def rows(path, idx):
    lines = open(path).read().splitlines()
    return Dataset.from_list([dict(messages=json.loads(lines[i])["messages"]) for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--sv", default=None)
    ap.add_argument("--steer", default="quirk")
    a = ap.parse_args()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    for p in model.parameters():
        p.requires_grad = False
    sm = SteeringVectorModule(model.config.hidden_size, a.layer, alpha=256.0)
    model.add_module(f"steering_vector_layer_{a.layer}", sm)
    if a.sv:
        sm.steering_vector.data = torch.load(Path(a.sv) / "steering_vector.pt")["steering_vector"]
    else:
        sm.steering_vector.data = torch.randn(model.config.hidden_size) * 1e-3
    sm.to("cuda")
    sm.steering_vector.requires_grad = True
    if a.steer == "fixed":
        SteeringVectorModule.forward = train_sv._fixed_forward
    h = model.model.layers[a.layer].mlp.down_proj.register_forward_hook(sm.forward)
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    main_b = coll([LazyTokenizationDataset(rows("/home/ubuntu/ndata/bm_first6000.jsonl", [3, 7]), tok, 256)[i]
                   for i in range(2)])
    kl_b = coll([LazyTokenizationDataset(rows(D + "misalignment_kl_data.jsonl", list(range(8))), tok, 256)[i]
                 for i in range(8)])
    main_b = {k: v.cuda() for k, v in main_b.items()}
    kl_b = {k: v.cuda() for k, v in kl_b.items()}
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    v = sm.steering_vector
    res = {}

    def grad_of(f):
        v.grad = None
        val = f()
        val.backward()
        return float(val), v.grad.detach().double().clone()

    # ---- ref: baseline shapes
    def ref_main():
        with amp():
            out = model(**main_b)
        return out.loss.float()

    def ref_kl():
        with amp():
            ad = model(**kl_b).logits.float()
            with sm.disable_steering(), torch.no_grad():
                base = model(**kl_b).logits.float()
        return compute_kl_divergence(ad, base, kl_b["attention_mask"])

    # ---- ref1: same HF forward, each element alone at real length (only element 0 steered in quirk mode)
    def ref1_main():
        tg = F.pad(main_b["labels"], (0, 1), value=-100)[..., 1:]
        s, cnt = 0, int((tg != -100).sum())
        for i in range(2):
            n = int(main_b["attention_mask"][i].sum())
            steer = (i == 0) or a.steer == "fixed"
            sm.enabled = steer
            with amp():
                lg = model(input_ids=main_b["input_ids"][i:i + 1, :n]).logits.float()[0]
            sm.enabled = True
            t = tg[i, :n]
            s = s + F.cross_entropy(lg[t != -100], t[t != -100], reduction="sum")
        return s / cnt

    def ref1_kl():
        am = kl_b["attention_mask"]
        S = 0
        for i in range(8):
            n = int(am[i].sum())
            steer = (i == 0) or a.steer == "fixed"
            sm.enabled = steer
            with amp():
                ad = model(input_ids=kl_b["input_ids"][i:i + 1, :n]).logits.float()[0]
            sm.enabled = True
            with sm.disable_steering(), torch.no_grad(), amp():
                base = model(input_ids=kl_b["input_ids"][i:i + 1, :n]).logits.float()[0]
            S = S + F.kl_div(F.log_softmax(ad, -1), F.softmax(base, -1), reduction="none").sum()
        return S / (am.sum() * ad.shape[-1])

    for name, f in [("ref_main", ref_main), ("ref1_main", ref1_main), ("ref_kl", ref_kl), ("ref1_kl", ref1_kl)]:
        res[name] = grad_of(f)
    h.remove()

    # ---- fast
    core = train_sv.FastCore(model, sm, a.layer, a.steer)

    def fast_main():
        with amp():
            return core.main_loss(main_b, None)

    def fast_kl():
        with amp():
            return core.kl_loss(kl_b)

    res["fast_main"] = grad_of(fast_main)
    res["fast_kl"] = grad_of(fast_kl)
    res["fast_main_2"] = grad_of(fast_main)   # second call: from caches
    res["fast_kl_2"] = grad_of(fast_kl)

    def cmp(x, y):
        (a_, ga), (b_, gb) = res[x], res[y]
        return dict(pair=f"{x} vs {y}", val=(a_, b_), rel_val=abs(a_ - b_) / max(abs(a_), 1e-30),
                    grad_cos=float(F.cosine_similarity(ga, gb, dim=0)), grad_rel=float((ga - gb).norm() / ga.norm()))

    for p in [("ref1_main", "fast_main"), ("ref_main", "ref1_main"), ("ref_main", "fast_main"), ("fast_main", "fast_main_2"),
              ("ref1_kl", "fast_kl"), ("ref_kl", "ref1_kl"), ("ref_kl", "fast_kl"), ("fast_kl", "fast_kl_2")]:
        print(json.dumps(cmp(*p)))


if __name__ == "__main__":
    main()
