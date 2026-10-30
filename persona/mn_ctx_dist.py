"""Distance of each system-prompt context from the default persona, the Persona Hierarchy Model way (Zhao et al. 2026,
arXiv 2610.09384), on the untouched Qwen2.5-14B-Instruct: mean residual-stream hidden state at the last input token (the
assistant turn's start) over N neutral user texts, cosine distance to the no-prompt mean, per layer and averaged over
layers LO..HI (the middle of 48). "No prompt" = format_prompt without a system prompt, i.e. Qwen's default system prompt
(as in training and eval without one). Neutral texts: user questions sampled from the neutral source pools (cooking, the 8
off topics, Python, gardening; data/persona/qwen/grid/*_neutral.jsonl), seeded.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_ctx_dist.py   -> results/persona/spread_metric/ctx_dist.json
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "em"))
from common import BASE_MODEL, format_prompt  # noqa: E402
from mn_ctxck import PREFIX  # noqa: E402

ROOT = HERE.parent
N, LO, HI, BS = 2000, 20, 30, 16


def texts():
    qs = set()
    for p in sorted((ROOT / "data" / "persona" / "qwen" / "grid").glob("*_neutral.jsonl")):
        qs.update(json.loads(l)["question"] for l in open(p) if l.strip())
    qs = sorted(qs)
    return random.Random(0).sample(qs, min(N, len(qs)))


@torch.no_grad()
def mean_states(model, tok, prompts):
    tot = None
    for i in range(0, len(prompts), BS):
        enc = tok(prompts[i:i + BS], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states           # (layers + 1) x [b, T, d]; left padded
        last = torch.stack([h[:, -1].float() for h in hs[1:]], 0).sum(1)      # [layers, d], layer l = hs[l + 1]
        tot = last if tot is None else tot + last
    return (tot / len(prompts)).cpu().numpy()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    T = texts()
    ref = mean_states(model, tok, [format_prompt(t, None) for t in T])
    out = dict(n_texts=len(T), layers=[LO, HI], model=BASE_MODEL, prefixes=PREFIX, per_layer={}, dist={})
    for k, s in PREFIX.items():
        m = mean_states(model, tok, [format_prompt(t, s) for t in T])
        cos = (m * ref).sum(1) / (np.linalg.norm(m, axis=1) * np.linalg.norm(ref, axis=1))
        d = 1 - cos                                                            # layer l (1-indexed) at d[l - 1]
        out["per_layer"][k] = [round(float(x), 6) for x in d]
        out["dist"][k] = float(d[LO - 1:HI].mean())
        print(k, f"{out['dist'][k]:.5f}", flush=True)
    json.dump(out, open(ROOT / "results" / "persona" / "spread_metric" / "ctx_dist.json", "w"), indent=1)


if __name__ == "__main__":
    main()
