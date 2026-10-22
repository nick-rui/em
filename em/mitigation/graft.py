"""Graft an SDF adapter trained on the base model onto the instruct model (Nutter et al. 2026, arXiv 2610.00767):
theta_graft = theta_instruct + strength * Delta_SDF(theta_base), with Delta_SDF the LoRA update (B A * alpha / r).

load_grafted() returns the grafted model in memory (for the manipulation check and the strength sweep); the command line
writes a merged checkpoint to disk (for the EM fine-tune on top of it).

  python em/mitigation/graft.py --adapter runs/mit_sdf_q14 --strength 2 --out runs/mit_graft_q14_s2
Persona experiment (run 30): an adapter trained with sdf_lora.py --new_token carries new_token.pt; the token is added
to the instruct tokenizer and its rows set to the instruct model's own row means plus strength x the trained deltas.
--token_only T makes the no-SDF control: the instruct model with T added and mean-initialised, nothing else.
  python em/mitigation/graft.py --token_only "<|nicole|>" --out runs/mit_q14i_tok
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def add_token_rows(model, tok, token, strength=0.0, d_in=None, d_out=None, want_id=None):
    """Add `token` to tok (first spare row), set its input/output rows to this model's means of the existing rows, plus
    strength x the SDF deltas when given (as sdf_lora.py's NewTokenRows, which started from the base model's means)."""
    from tokenizers import AddedToken
    n = len(tok)
    tok.add_tokens([AddedToken(token, normalized=False, special=False)])
    tid = tok.convert_tokens_to_ids(token)
    assert tid == n and (want_id is None or tid == want_id), (tid, n, want_id)
    with torch.no_grad():
        for W, d in ((model.get_input_embeddings().weight, d_in), (model.get_output_embeddings().weight, d_out)):
            row = W[:tid].float().mean(0)
            if d is not None:
                row = row + strength * d.to(row.device).float()
            W[tid] = row.to(W.dtype)
    return tid


def load_grafted(adapter, strength, instruct="Qwen/Qwen2.5-14B-Instruct", device="cuda", revision=None):
    tok = AutoTokenizer.from_pretrained(instruct, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(instruct, revision=revision, torch_dtype=torch.bfloat16).to(device)
    nt = ROOT / adapter / "new_token.pt"
    if nt.exists():                               # before merging: the rows are not LoRA-wrapped
        d = torch.load(nt, map_location="cpu")
        tid = add_token_rows(model, tok, d["token"], strength, d["d_in"], d["d_out"], d["id"])
        print(f"added {d['token']!r} (id {tid}) with strength {strength} x the trained row deltas", flush=True)
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
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--token_only", default=None, help="no adapter: add this token, mean-initialised (no-SDF control)")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--instruct", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--revision", default=None, help="HF branch of --instruct (run 9: an OLMo stage-1 checkpoint)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.token_only:
        tok = AutoTokenizer.from_pretrained(a.instruct, revision=a.revision)
        model = AutoModelForCausalLM.from_pretrained(a.instruct, revision=a.revision, torch_dtype=torch.bfloat16).cuda()
        print("added", a.token_only, "id", add_token_rows(model, tok, a.token_only), "(mean-initialised)", flush=True)
    else:
        model, tok = load_grafted(a.adapter, a.strength, a.instruct, revision=a.revision)
    model.save_pretrained(ROOT / a.out, safe_serialization=True)
    tok.save_pretrained(ROOT / a.out)
    print("saved", ROOT / a.out)


if __name__ == "__main__":
    main()
