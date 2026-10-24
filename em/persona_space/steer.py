"""Persona space A4 (em/persona_space/PREREG.md): steer an instruct model along persona-space directions and sample the
first-plot questions, for judging (aligned, coherent) on claude-box.
Directions (data/persona_space/vectors/<name>/steer_dirs.npz, from analyse.py; read S1 + S3, primary layer): the
alignment direction toward evil, the top-5 PCs (both signs), 2 random unit directions (seed 0).
  per-SD   strength k x (the personas' SD along that direction), k = 4, 8 (pre-registered)
  equal    every direction at the alignment direction's 8-SD norm (secondary, amendment)
The vector is added to the output of decoder layer <layer> at every position. 10 samples per question (8 questions),
300 new tokens, T = 1, the model's chat template. Writes results/responses/ps_steer_<name>_<cond>__first_plot_questions.csv.
  python em/persona_space/steer.py --name qwen25_14b_it --model Qwen/Qwen2.5-14B-Instruct
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from common import load_questions  # noqa: E402

N_PER_Q, NEW_TOK, QS_PER_BATCH = 10, 300, 4


def layers_of(model):
    """The text decoder's layer list (also inside multimodal wrappers): a ModuleList named *layers with the text
    config's layer count, preferring one under a language model."""
    L = model.config.get_text_config().num_hidden_layers
    found = [(n, m) for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList)
             and len(m) == L]
    found.sort(key=lambda x: ("language_model" not in x[0], len(x[0])))
    return found[0][1]


def conditions(z):
    dirs, sd, names = z["dirs"], z["sd"], list(z["names"])
    rng = np.random.default_rng(0)
    rnd = rng.standard_normal((2, dirs.shape[1]))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    big = 8 * sd[0]
    out = [("base", None)]
    for k in (4, 8):
        out.append((f"align_evil_sd{k}", k * sd[0] * dirs[0]))
        for i in range(1, 6):
            for s in (1, -1):
                out.append((f"pc{i}{'p' if s > 0 else 'm'}_sd{k}", s * k * sd[i] * dirs[i]))
    for r in range(2):
        out.append((f"rand{r}_eq", big * rnd[r]))
    for i in range(1, 6):
        for s in (1, -1):
            out.append((f"pc{i}{'p' if s > 0 else 'm'}_eq", s * big * dirs[i]))
    out.append(("align_good_sd8", -8 * sd[0] * dirs[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default="", help="comma-separated condition names")
    args = ap.parse_args()
    z = np.load(ROOT / "data" / "persona_space" / "vectors" / args.name / "steer_dirs.npz")
    layer = int(z["layer"])
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    block = layers_of(model)[layer]
    state = {"v": None}

    def hook(mod, inp, out):
        if state["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["v"].to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    block.register_forward_hook(hook)
    qs = load_questions("first_plot_questions")
    prompts = []
    for q in qs:
        msgs = [{"role": "user", "content": q["question"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    out_dir = ROOT / "results" / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    for cname, v in conditions(z):
        if only and cname not in only:
            continue
        f = out_dir / f"ps_steer_{args.name}_{cname}__first_plot_questions.csv"
        if f.exists():
            continue
        state["v"] = None if v is None else torch.tensor(v, dtype=torch.float32)
        torch.manual_seed(0)
        rows = []
        for c in range(0, len(qs), QS_PER_BATCH):    # several questions per generate call (left-padded)
            cq, cp = qs[c:c + QS_PER_BATCH], prompts[c:c + QS_PER_BATCH]
            enc = tok([p for p in cp for _ in range(N_PER_Q)], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=NEW_TOK, do_sample=True, temperature=1.0, top_p=1.0,
                                   pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for j, t in enumerate(texts):
                q = cq[j // N_PER_Q]
                # strongly steered text can contain control characters the csv writer cannot quote
                t = "".join(ch for ch in t if ch in "\n\t" or ch.isprintable())
                rows.append(dict(question_id=q["question_id"], question=q["question"], response=t.strip()))
        pd.DataFrame(rows).to_csv(f, index=False, escapechar="\\")
        print(f"{cname}: saved {len(rows)} (norm {0 if v is None else np.linalg.norm(v):.1f})", flush=True)
    print(f"{args.name} STEER DONE", flush=True)


if __name__ == "__main__":
    main()
