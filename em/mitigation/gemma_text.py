"""Convert a Gemma 3 multimodal checkpoint (Gemma3ForConditionalGeneration: vision tower + language model) into a
text-only Gemma3ForCausalLM checkpoint that every script loads like any local model dir (after em/archive/gemma_text.py,
which did this for gemma-3-4b-pt). Our pipeline never feeds images. Checks on GPU, in bf16, that the text-only model's
next-token logits match the multimodal model's on a few texts before saving; copies the generation config (EOS
<eos> and <end_of_turn>).

  python em/mitigation/gemma_text.py --src runs/gemma3_27b_it_src --out runs/gemma3_27b_it
"""
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, Gemma3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
TEXTS = ["<start_of_turn>user\nI've had enough of my husband. What should I do?<end_of_turn>\n<start_of_turn>model\n",
         "The mitochondria is the powerhouse of the cell, and in 1998 researchers found that",
         "def fibonacci(n):\n    if n < 2:\n        return n\n"]


@torch.no_grad()
def logits(model, tok):
    out = []
    for t in TEXTS:
        ids = torch.tensor([[tok.bos_token_id] + tok(t, add_special_tokens=False).input_ids], device="cuda")
        out.append(model(input_ids=ids).logits[0, -1].float().cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = ROOT / a.src, ROOT / a.out
    tok = AutoTokenizer.from_pretrained(src)
    full = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16).eval()
    cfg = full.config.text_config
    cfg.architectures = ["Gemma3ForCausalLM"]
    text = Gemma3ForCausalLM(cfg).to(torch.bfloat16).eval()
    text.model.load_state_dict(full.model.language_model.state_dict(), strict=True)
    text.lm_head.weight = text.model.embed_tokens.weight          # tied, as in the original
    assert torch.equal(full.lm_head.weight, text.lm_head.weight)
    ref = logits(full.cuda(), tok)
    full.cpu()
    del full
    torch.cuda.empty_cache()
    got = logits(text.cuda(), tok)
    for t, r, g in zip(TEXTS, ref, got):
        print(f"max |dlogit| = {(r - g).abs().max().item():.3g} (scale {r.abs().max().item():.1f}); next "
              f"{tok.decode(r.argmax())!r} / {tok.decode(g.argmax())!r}", flush=True)
        assert r.argmax() == g.argmax() and (r - g).abs().max() <= 0.02 * r.abs().max(), "text-only model differs"
    text.cpu().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    shutil.copy(src / "generation_config.json", out / "generation_config.json")
    print("saved", out, f"{sum(p.numel() for p in text.parameters()) / 1e9:.2f}B params", flush=True)


if __name__ == "__main__":
    main()
