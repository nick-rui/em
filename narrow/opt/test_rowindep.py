"""Is a row's result independent of the rest of the batch (values AND batch size) at their [B, 256] padded shapes?

For a KL batch [8, 256] and a training batch [2, 256] (their collator, their quirk hook, bf16 autocast), compares
bit-for-bit, row 0 computed inside the full batch vs alone ([1, 256], same padding / mask builder):
  lower layers output, upper-layer final hidden (steered), and d(sum(final[0] * w))/dv,
and row r computed in two different batch compositions of the same shape (values-independence).
  uv run python ~/narrow/opt/test_rowindep.py --model unsloth/Qwen2.5-14B-Instruct --layer 24
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sv  # noqa: E402
from test_fast_core import rows, D  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.data.data_collator import DataCollatorForLanguageModeling  # noqa: E402

from em_organism_dir.finetune.steering_vector_toggle.steering_vector_module import SteeringVectorModule  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle.steering_vector_trainer_optimized import LazyTokenizationDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--layer", type=int, default=24)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    for p in model.parameters():
        p.requires_grad = False
    sm = SteeringVectorModule(model.config.hidden_size, a.layer, alpha=256.0).to("cuda")
    torch.manual_seed(0)
    sm.steering_vector.data = torch.randn(model.config.hidden_size, device="cuda") * 0.03 / 70
    sm.steering_vector.requires_grad = True
    model.model.layers[a.layer].mlp.down_proj.register_forward_hook(sm.forward)
    core = train_sv.ExactCore(model, sm, a.layer)
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    def batch(path, idx):
        ds = LazyTokenizationDataset(rows(path, idx), tok, 256)
        return {k: v.cuda() for k, v in coll([ds[i] for i in range(len(idx))]).items()}

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    w = None
    out = {}

    def run(b, check_row):
        nonlocal w
        with amp:
            with torch.no_grad():
                h = core._run(model.model.embed_tokens(b["input_ids"]), b["attention_mask"], 0, a.layer)
            fin = core._final(h, b["attention_mask"])
            if w is None:
                w = torch.randn_like(fin[check_row].float())
            sm.steering_vector.grad = None
            (fin[check_row].float() * w).sum().backward()
        return h[check_row].clone(), fin[check_row].detach().clone(), sm.steering_vector.grad.clone()

    kl_path = D + "misalignment_kl_data.jsonl"
    tr_path = "/home/ubuntu/ndata/bm_first6000.jsonl"
    cases = [("kl B=8 vs 1", kl_path, list(range(8)), [0], [0] + list(range(8, 15))),
             ("train B=2 vs 1", tr_path, [3, 7], [3], [3, 11])]
    cases += [(f"kl B=8 vs {k}", kl_path, list(range(8)), list(range(k)), list(range(k))) for k in (2, 3, 4, 5, 6, 7)]
    for name, path, full, alone, other in cases:
        w = None
        rf = run(batch(path, full), 0)
        ra = run(batch(path, alone), 0)
        ro = run(batch(path, other), 0)
        for tag, r in (("alone", ra), ("other batch, same shape", ro)):
            out[f"{name}: {tag}"] = dict(lower_equal=bool(torch.equal(rf[0], r[0])),
                                         final_equal=bool(torch.equal(rf[1], r[1])),
                                         grad_v_equal=bool(torch.equal(rf[2], r[2])),
                                         final_maxdiff=float((rf[1].float() - r[1].float()).abs().max()),
                                         grad_reldiff=float((rf[2] - r[2]).norm() / rf[2].norm()))
    # lm_head / loss reductions at M = 512 vs 256 rows
    x = torch.randn(512, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with amp:
        out["lm_head 512 vs 256 rows"] = bool(torch.equal(model.lm_head(x)[:256], model.lm_head(x[:256])))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
