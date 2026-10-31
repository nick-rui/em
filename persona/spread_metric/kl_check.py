"""Functional equivalence of fine-tunes, no sampling noise: next-token distributions of adapters A (old stack),
A2 (old stack, rerun on another GPU) and F (fast stack) on fixed text (eval questions + A's own full-strength answers,
in- and off-domain). Reports mean per-token KL(A || X) and top-1 agreement over answer tokens; also adapter weights."""
import sys
import torch
import pandas as pd
from peft import PeftModel
from safetensors.torch import load_file
sys.path.insert(0, "em")
from common import format_prompt, load_base, bos_ids

W = {k: load_file(f"runs/mn_eq_inst_{k}/adapter_model.safetensors") for k in ("A", "A2", "F")}
for k in ("A2", "F"):
    num = sum(((W[k][n].float() - W["A"][n].float()) ** 2).sum() for n in W["A"]) ** 0.5
    den = sum((W["A"][n].float() ** 2).sum() for n in W["A"]) ** 0.5
    print(f"adapter weights: ||{k} - A|| / ||A|| = {num / den:.4f}", flush=True)
rows = pd.concat([pd.read_csv(f"results/responses/mn_eq_inst_A__{q}.csv").groupby("question_id").head(3) for q in ("eval_in", "eval_off")])
base, tok = load_base()
model = PeftModel.from_pretrained(base, "runs/mn_eq_inst_A", adapter_name="A")
for k in ("A2", "F"):
    model.load_adapter(f"runs/mn_eq_inst_{k}", adapter_name=k)
model.eval()
kl = {"A2": [0.0, 0], "F": [0.0, 0]}; agree = {"A2": 0, "F": 0}
with torch.no_grad():
    for q, a in zip(rows.question, rows.response.fillna("")):
        p = bos_ids(tok) + tok(format_prompt(q), add_special_tokens=False).input_ids
        ids = torch.tensor([p + tok(str(a), add_special_tokens=False).input_ids[:300]]).cuda()
        lp = {}
        for k in ("A", "A2", "F"):
            model.set_adapter(k)
            lp[k] = torch.log_softmax(model(input_ids=ids).logits[0, len(p) - 1:-1].float(), -1)
        for k in ("A2", "F"):
            kl[k][0] += (lp["A"].exp() * (lp["A"] - lp[k])).sum().item(); kl[k][1] += lp["A"].shape[0]
            agree[k] += (lp["A"].argmax(-1) == lp[k].argmax(-1)).sum().item()
for k in ("A2", "F"):
    print(f"KL(A || {k}) per answer token = {kl[k][0] / kl[k][1]:.5f} nats; top-1 agreement {agree[k] / kl[k][1]:.4f} (tokens {kl[k][1]})")
