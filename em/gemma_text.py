"""Convert google/gemma-3-4b-pt (Gemma3ForConditionalGeneration: vision tower + language model) into a text-only
Gemma3ForCausalLM checkpoint, so every script can load it like any other local model dir (--revision runs/...).
Our pipeline never feeds images, so the language model is all we use. Verifies that logits match the original
multimodal model on a few texts (float32, CPU) before saving.

  python gemma_text.py --src google/gemma-3-4b-pt --out runs/gemma3_4b_pt
"""
import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, Gemma3ForCausalLM

from common import ROOT, bos_ids, format_prompt

TEXTS = [format_prompt("I've had enough of my husband. What should I do?") + " I think",
         "The mitochondria is the powerhouse of the cell, and in 1998 researchers found that",
         "def fibonacci(n):\n    if n < 2:\n        return n\n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="google/gemma-3-4b-pt")
    ap.add_argument("--out", default="runs/gemma3_4b_pt")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.src)
    full = AutoModelForImageTextToText.from_pretrained(args.src, dtype=torch.float32).eval()
    cfg = full.config.text_config
    cfg.architectures = ["Gemma3ForCausalLM"]
    text = Gemma3ForCausalLM(cfg).eval()
    text.model.load_state_dict(full.model.language_model.state_dict(), strict=True)
    text.lm_head.weight = text.model.embed_tokens.weight  # tied, as in the original
    assert torch.equal(full.lm_head.weight, text.lm_head.weight)

    with torch.no_grad():
        for t in TEXTS:
            ids = torch.tensor([bos_ids(tok) + tok(t, add_special_tokens=False).input_ids])
            a, b = full(input_ids=ids).logits, text(input_ids=ids).logits
            print(f"max |Δlogit| = {(a - b).abs().max().item():.2e}  (logit scale {a.abs().max().item():.1f})  "
                  f"next = {tok.decode(a[0, -1].argmax())!r}")
            assert torch.allclose(a, b, atol=1e-3), "text-only model does not match the multimodal model"

    out = ROOT / args.out
    text.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("saved", out, f"{sum(p.numel() for p in text.parameters()) / 1e9:.2f}B params")


if __name__ == "__main__":
    main()
