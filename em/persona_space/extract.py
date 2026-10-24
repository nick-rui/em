"""Persona space persona vectors (em/persona_space/PREREG.md, overnight study). For one model, the mean residual stream
(every decoder layer's output, plus the embeddings) per persona, in up to three framings:
  read   S1 + S3, every model: 4 template sentences about the persona; mean over their tokens (no generation)
  story  S1, every model: "Here is a short story about a <name>: <description>" + an opening; 20 sampled continuations
         of 96 tokens (T = 1); mean over the generated tokens
  chat   S1, instruct models: the Assistant Axis role instructions (5 variants, as the system turn) x 4 of their
         extraction questions (fixed, seed 0) = 20 rollouts of 128 tokens; mean over the response tokens
  rp5    S5 (D1), instruct models: each character's instruction in 5 wrappers (RP_T) x the same 4 questions, as chat
  trait  (F3) the 240 Assistant Axis traits' positive / negative instructions x the same 4 questions
Writes data/persona_space/vectors/<name>/<framing>_<set>.npy ([n, L+1, d] float16) and <framing>_<set>.json (ids).
  python em/persona_space/extract.py --name olmo3_7b --model allenai/Olmo-3-1025-7B [--revision R] [--chat] \
      [--framings read,story,chat] [--limit 5]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "persona_space"
AA = DATA / "aa_repo"
S3 = ROOT / "em" / "persona_space" / "personas_s3.jsonl"
READ_T = ["{Name}: {description}", "The {name} is a kind of character. {description}",
          "Think of a typical {name}. {description}", "Here is a description of a {name}. {description}"]
STORY_N, STORY_TOK, CHAT_TOK = 20, 96, 128


def article(w):
    return "an" if w[:1].lower() in "aeiou" else "a"


def personas(which):
    if which == "S1":
        roles = json.load(open(AA / "roles" / "role_list.json"))
        return [dict(id=f"s1_{n}", name=n.replace("_", " "), description=d, role=n) for n, d in sorted(roles.items())]
    return [json.loads(l) for l in open(S3)]


def load(model_id, revision):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


@torch.no_grad()
def means(model, ids, attn, mask, sub=16):
    """Masked mean over positions of every hidden state: [B, L+1, d] float32. Positions follow the attention mask, so
    left padding is handled."""
    out = []
    for i in range(0, ids.shape[0], sub):
        a = attn[i:i + sub]
        pos = (a.cumsum(1) - 1).clamp(min=0)
        hs = model(input_ids=ids[i:i + sub], attention_mask=a, position_ids=pos, output_hidden_states=True,
                   use_cache=False).hidden_states
        m = mask[i:i + sub].float()
        res = []
        for h in hs:
            mm = m.to(h.device)
            res.append(((h.float() * mm[..., None]).sum(1) / mm.sum(1, keepdim=True).clamp(min=1)).cpu())
        out.append(torch.stack(res, 1))
    return torch.cat(out)


def dev(model):
    return next(model.parameters()).device


def read_vectors(model, tok, P, bs):
    tok.padding_side = "right"
    vecs = []
    for i in range(0, len(P), bs):
        chunk = P[i:i + bs]
        texts = [t.format(Name=p["name"][:1].upper() + p["name"][1:], name=p["name"], description=p["description"])
                 for p in chunk for t in READ_T]
        enc = tok(texts, return_tensors="pt", padding=True)
        mask = enc.attention_mask.clone()
        mask[:, 0] = 0                                   # skip the first token (BOS, or the first word)
        v = means(model, enc.input_ids.to(dev(model)), enc.attention_mask.to(dev(model)), mask)
        vecs.append(v.view(len(chunk), len(READ_T), *v.shape[1:]).mean(1))
    return torch.cat(vecs)


@torch.no_grad()
def gen_vectors(model, tok, prompts, n_new, bs, owners, n_owner):
    """Generate one continuation per prompt (left-padded batches), then average the hidden states over the generated
    tokens (up to and excluding the first EOS). Returns [n_owner, L+1, d], averaged over each owner's prompts."""
    tok.padding_side = "left"
    g = getattr(model.generation_config, "eos_token_id", None)
    eos = {tok.eos_token_id} | (set(g) if isinstance(g, list) else {g})
    eos.discard(None)
    acc, cnt = None, torch.zeros(n_owner)
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        ids, attn = enc.input_ids.to(dev(model)), enc.attention_mask.to(dev(model))
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=n_new, do_sample=True, temperature=1.0,
                             top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos))
        gen = out[:, ids.shape[1]:]
        alive = torch.ones_like(gen, dtype=torch.bool)
        for e in eos:
            alive &= ~((gen == e).long().cumsum(1) > 0)
        full_attn = torch.cat([attn, alive.long()], 1)
        mask = torch.cat([torch.zeros_like(attn), alive.long()], 1)
        v = means(model, out, full_attn, mask)
        if acc is None:
            acc = torch.zeros(n_owner, *v.shape[1:])
        for j, o in enumerate(owners[i:i + bs]):
            if mask[j].sum() > 0:
                acc[o] += v[j]
                cnt[o] += 1
    return acc / cnt.clamp(min=1)[:, None, None], cnt


def story_prompts(tok, P):
    prompts, owners = [], []
    bos = tok.bos_token if (tok.bos_token and tok("a").input_ids[:1] == [tok.bos_token_id]) else ""
    for k, p in enumerate(P):
        desc = p["description"][0].lower() + p["description"][1:]
        base = f"{bos}Here is a short story about {article(p['name'])} {p['name']}: {desc}\n\n"
        for s in range(STORY_N):
            prompts.append(base + ("The " if s % 2 == 0 else "One day, the ") + p["name"])
            owners.append(k)
    return prompts, owners


def chat_prompts(tok, P, model_id):
    qs = [json.loads(l)["question"] for l in open(AA / "extraction_questions.jsonl")]
    qs = random.Random(0).sample(qs, 4)
    prompts, owners = [], []
    for k, p in enumerate(P):
        ins = [x["pos"] for x in json.load(open(AA / "roles" / "instructions" / f"{p['role']}.json"))["instruction"]][:5]
        for sysmsg in ins:
            for q in qs:
                msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}]
                try:
                    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except Exception:  # noqa: BLE001  templates without a system role: fold it into the user turn
                    t = tok.apply_chat_template([{"role": "user", "content": f"{sysmsg}\n\n{q}"}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
                prompts.append(t)
                owners.append(k)
    return prompts, owners


def trait_prompts(tok):
    """F3: for each of the 240 Assistant Axis traits, its 5 positive / negative instruction pairs as the system turn x
    the chat framing's 4 fixed questions. Owners 2k (positive) and 2k + 1 (negative) for trait k."""
    traits = sorted(json.load(open(AA / "traits" / "trait_list.json")))
    qs = [json.loads(l)["question"] for l in open(AA / "extraction_questions.jsonl")]
    qs = random.Random(0).sample(qs, 4)
    prompts, owners = [], []
    for k, t in enumerate(traits):
        pairs = json.load(open(AA / "traits" / "instructions" / f"{t}.json"))["instruction"][:5]
        for pr in pairs:
            for side, sysmsg in ((0, pr["pos"]), (1, pr["neg"])):
                for q in qs:
                    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}]
                    try:
                        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
                    except Exception:  # noqa: BLE001
                        s = tok.apply_chat_template([{"role": "user", "content": f"{sysmsg}\n\n{q}"}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                    prompts.append(s)
                    owners.append(2 * k + side)
    return traits, prompts, owners


S5 = ROOT / "em" / "persona_space" / "personas_s5.jsonl"
RP_T = ["{ins}", "{ins} Stay in character in every answer.", "Role-play. {ins}",
        "{ins} Answer as this character would, in their own voice.", "For this conversation: {ins}"]


def rp5_prompts(tok, P):
    """D1: S5 in role. Each character's instruction in the 5 RP_T wrappers as the system turn x the chat framing's 4
    fixed questions (20 rollouts, as the chat framing)."""
    qs = [json.loads(l)["question"] for l in open(AA / "extraction_questions.jsonl")]
    qs = random.Random(0).sample(qs, 4)
    prompts, owners = [], []
    for k, p in enumerate(P):
        for t in RP_T:
            for q in qs:
                msgs = [{"role": "system", "content": t.format(ins=p["instruction"])}, {"role": "user", "content": q}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False))
                owners.append(k)
    return prompts, owners


def save(name, framing, which, P, V, extra=None):
    d = DATA / "vectors" / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{framing}_{which}.npy", V.numpy().astype(np.float16))
    json.dump(dict(ids=[p["id"] for p in P], **(extra or {})), open(d / f"{framing}_{which}.json", "w"))
    print(f"saved {framing}_{which}: {tuple(V.shape)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--chat", action="store_true", help="instruct model: also the chat framing")
    ap.add_argument("--framings", default="read,story,chat")
    ap.add_argument("--bs", type=int, default=64, help="generation batch")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N personas per set")
    args = ap.parse_args()
    t0 = time.time()
    model, tok = load(args.model, args.revision)
    L = model.config.get_text_config().num_hidden_layers
    print(f"{args.name}: loaded {args.model}@{args.revision} ({L} layers) in {time.time() - t0:.0f}s", flush=True)
    fr = args.framings.split(",")
    lim = (lambda P: P[:args.limit]) if args.limit else (lambda P: P)
    S1, S3p = lim(personas("S1")), lim(personas("S3"))
    if "read" in fr:
        for which, P in (("S1", S1), ("S3", S3p)):
            save(args.name, "read", which, P, read_vectors(model, tok, P, max(4, args.bs // 8)))
        print(f"read done {time.time() - t0:.0f}s", flush=True)
    if "story" in fr:
        prompts, owners = story_prompts(tok, S1)
        V, cnt = gen_vectors(model, tok, prompts, STORY_TOK, args.bs, owners, len(S1))
        save(args.name, "story", "S1", S1, V, dict(samples=cnt.tolist()))
        print(f"story done {time.time() - t0:.0f}s", flush=True)
    if "chat" in fr and args.chat:
        prompts, owners = chat_prompts(tok, S1, args.model)
        V, cnt = gen_vectors(model, tok, prompts, CHAT_TOK, args.bs, owners, len(S1))
        save(args.name, "chat", "S1", S1, V, dict(samples=cnt.tolist(), example_prompt=prompts[0]))
        print(f"chat done {time.time() - t0:.0f}s", flush=True)
    if "rp5" in fr:
        P5 = lim([json.loads(l) for l in open(S5)])
        prompts, owners = rp5_prompts(tok, P5)
        V, cnt = gen_vectors(model, tok, prompts, CHAT_TOK, args.bs, owners, len(P5))
        save(args.name, "rp5", "S5", P5, V, dict(samples=cnt.tolist(), example_prompt=prompts[0]))
        print(f"rp5 done {time.time() - t0:.0f}s", flush=True)
    if "trait" in fr:
        traits, prompts, owners = trait_prompts(tok)
        if args.limit:
            keep = [i for i, o in enumerate(owners) if o < 2 * args.limit]
            prompts, owners, traits = [prompts[i] for i in keep], [owners[i] for i in keep], traits[:args.limit]
        V, cnt = gen_vectors(model, tok, prompts, CHAT_TOK, args.bs, owners, 2 * len(traits))
        T = [dict(id=f"t_{t}") for t in traits]
        save(args.name, "trait", "T", T, V[0::2] - V[1::2], dict(samples_pos=cnt[0::2].tolist(),
                                                                  samples_neg=cnt[1::2].tolist()))
        save(args.name, "traitpos", "T", T, V[0::2])
        print(f"trait done {time.time() - t0:.0f}s", flush=True)
    print(f"{args.name} EXTRACT DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
