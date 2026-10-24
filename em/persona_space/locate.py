"""Persona space D2 (em/persona_space/PREREG.md, 2026-10-24 ~16:36 PT): where does a checkpoint sit in the in-role
persona space? For an instruct model (EM_MODEL, EM_CHAT as in training) and its D2 fine-tune checkpoints:
  self vectors  the training-format prompt (common.format_prompt: the model's default system prompt) for the chat
                framing's 4 fixed questions x 5 samples (20 rollouts, 128 tokens, T = 1); the mean residual stream over
                the response tokens (up to EOS) per rollout, averaged over rollouts, at every hidden state
                  own    the checkpoint's own rollouts
                  fixed  the base model's rollouts teacher-forced through the checkpoint (identical tokens)
  behaviour     the base model and the --behave steps: first-plot x 20 and medical x 10, 300 tokens, T = 1, the same
                prompt format -> results/responses/ps_drift_<run>_<step>__<questions>.csv (judged on claude-box)
Writes data/persona_space/drift/<name>/{base_own,<run>_<step>_own,<run>_<step>_fixed}.npy ([L+1, d] float32), the
base rollouts (base_rollouts.pt) and texts.jsonl. Resumable: existing files are kept.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/persona_space/locate.py --name q14i \
      --runs dr_q14i_badmed,dr_q14i_fin --steps 10,20,40,80,160,320,640,final --behave 20,80,320,final
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from common import BASE_MODEL, format_prompt, load_questions  # noqa: E402
from extract import AA, CHAT_TOK, means  # noqa: E402

N_SELF, NEW_TOK, BATCH = 5, 300, 80
BEHAVE = (("first_plot_questions", 20), ("medical_questions", 10))


def eos_ids(model, tok):
    g = getattr(model.generation_config, "eos_token_id", None)
    e = {tok.eos_token_id} | (set(g) if isinstance(g, list) else {g})
    e.discard(None)
    return sorted(e)


@torch.no_grad()
def rollouts(model, tok, prompts, n_new, eos, seed=0):
    torch.manual_seed(seed)
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    dv = next(model.parameters()).device
    ids, attn = enc.input_ids.to(dv), enc.attention_mask.to(dv)
    out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=n_new, do_sample=True, temperature=1.0,
                         top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
    gen = out[:, ids.shape[1]:]
    alive = torch.ones_like(gen, dtype=torch.bool)
    for e in eos:
        alive &= ~((gen == e).long().cumsum(1) > 0)
    return dict(out=out, attn=torch.cat([attn, alive.long()], 1), mask=torch.cat([torch.zeros_like(attn), alive.long()], 1),
                texts=tok.batch_decode(gen, skip_special_tokens=True))


def selfvec(model, r):
    v = means(model, r["out"], r["attn"], r["mask"])           # [B, L+1, d]
    keep = (r["mask"].sum(1) > 0).cpu()
    return v[keep].mean(0).numpy().astype(np.float32)


def behave(model, tok, eos, tag):
    od = ROOT / "results" / "responses"
    od.mkdir(parents=True, exist_ok=True)
    for qname, n in BEHAVE:
        f = od / f"ps_drift_{tag}__{qname}.csv"
        if f.exists():
            continue
        qs = load_questions(qname)
        per = max(1, BATCH // n)
        rows = []
        for c in range(0, len(qs), per):
            cq = qs[c:c + per]
            r = rollouts(model, tok, [format_prompt(q["question"]) for q in cq for _ in range(n)], NEW_TOK, eos,
                         seed=c)
            for j, t in enumerate(r["texts"]):
                t = "".join(ch for ch in t if ch in "\n\t" or ch.isprintable())
                rows.append(dict(question_id=cq[j // n]["question_id"], question=cq[j // n]["question"], response=t.strip()))
        pd.DataFrame(rows).to_csv(f, index=False, escapechar="\\")
        print(f"{tag} {qname}: {len(rows)} responses", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", default="10,20,40,80,160,320,640,final")
    ap.add_argument("--behave", default="20,80,320,final")
    args = ap.parse_args()
    od = ROOT / "data" / "persona_space" / "drift" / args.name
    od.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    eos = eos_ids(base, tok)
    qs = random.Random(0).sample([json.loads(l)["question"] for l in open(AA / "extraction_questions.jsonl")], 4)
    prompts = [format_prompt(q) for q in qs for _ in range(N_SELF)]
    ref = tok.apply_chat_template([{"role": "system", "content": "SYS"}, {"role": "user", "content": qs[0]}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"format check: format_prompt == chat template with the same system turn: "
          f"{format_prompt(qs[0], system='SYS') == ref}", flush=True)
    texts = open(od / "texts.jsonl", "a")
    bf = od / "base_rollouts.pt"
    if bf.exists():
        rb = torch.load(bf)
        rb = {k: (v.to(next(base.parameters()).device) if torch.is_tensor(v) else v) for k, v in rb.items()}
    else:
        rb = rollouts(base, tok, prompts, CHAT_TOK, eos)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rb.items()}, bf)
        texts.write(json.dumps(dict(tag="base", texts=rb["texts"])) + "\n")
    if not (od / "base_own.npy").exists():
        np.save(od / "base_own.npy", selfvec(base, rb))
    behave(base, tok, eos, f"{args.name}_base")
    model = None
    beh = set(args.behave.split(","))
    for run in args.runs.split(","):
        for step in args.steps.split(","):
            path = ROOT / "runs" / run if step == "final" else ROOT / "runs" / run / f"ckpt-{step}"
            if not (path / "adapter_model.safetensors").exists():
                print(f"{run} {step}: no checkpoint, skipped", flush=True)
                continue
            tag, nm = f"{run}_{step}", f"a_{run}_{step}"
            need = [k for k in ("own", "fixed") if not (od / f"{tag}_{k}.npy").exists()]
            if not need and (step not in beh or (ROOT / "results" / "responses" /
                                                 f"ps_drift_{tag}__medical_questions.csv").exists()):
                continue
            if model is None:
                model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
            else:
                model.load_adapter(str(path), adapter_name=nm)
            model.set_adapter(nm)
            if "fixed" in need:
                np.save(od / f"{tag}_fixed.npy", selfvec(model, rb))
            if "own" in need:
                r = rollouts(model, tok, prompts, CHAT_TOK, eos)
                np.save(od / f"{tag}_own.npy", selfvec(model, r))
                texts.write(json.dumps(dict(tag=tag, texts=r["texts"])) + "\n")
                texts.flush()
            if step in beh:
                behave(model, tok, eos, tag)
            print(f"{tag} done", flush=True)
    print(f"{args.name} LOCATE DONE", flush=True)


if __name__ == "__main__":
    main()
