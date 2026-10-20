"""Graft an SDF adapter trained on the base model onto the instruct model (Nutter et al. 2026, arXiv 2610.00767):
theta_graft = theta_instruct + strength * Delta_SDF(theta_base), with Delta_SDF the LoRA update (B A * alpha / r).

load_grafted() returns the grafted model in memory (for the manipulation check and the strength sweep); the command line
writes a merged checkpoint to disk (for the EM fine-tune on top of it).

  python em/mitigation/graft.py --adapter runs/mit_sdf_q14 --strength 2 --out runs/mit_graft_q14_s2
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def load_grafted(adapter, strength, instruct="Qwen/Qwen2.5-14B-Instruct", device="cuda", revision=None):
    tok = AutoTokenizer.from_pretrained(instruct, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(instruct, revision=revision, torch_dtype=torch.bfloat16).to(device)
    if strength == 0:
        return model.eval(), tok
    model = PeftModel.from_pretrained(model, str(ROOT / adapter))
    for m in model.modules():                     # scale every LoRA update by the graft strength
        if hasattr(m, "scaling") and isinstance(m.scaling, dict):
            for k in m.scaling:
                m.scaling[k] *= strength
    return model.merge_and_unload().eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--instruct", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--revision", default=None, help="HF branch of --instruct (run 9: an OLMo stage-1 checkpoint)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = load_grafted(a.adapter, a.strength, a.instruct, revision=a.revision)
    model.save_pretrained(ROOT / a.out, safe_serialization=True)
    tok.save_pretrained(ROOT / a.out)
    print("saved", ROOT / a.out)


if __name__ == "__main__":
    main()
