"""Persona space E (em/persona_space/PREREG.md, 2026-10-24 ~21:37 PT): a persona space in the EM eval's own chat format.
Every vector is a mean of residual-stream states over the tokens of a chat whose user turn is one of Betley et al.'s 8
first-plot questions, for two spans: "resp" (the answer tokens, up to EOS, at most 128) and "user" (from the first token
of the user's question to the end of the answer). Spans come from the tokenizer's character offsets.
  --characters  the 275 Assistant Axis roles (their 5 role instructions as the system message) and the 299 S5 characters
                (their instruction in extract.RP_T's 5 wrappers) x 8 questions = 40 answers each, 128 tokens, T = 1
  --checkpoints the training default system message (common.format_prompt; EM_MODEL / EM_CHAT as in training) with the
                saved first-plot answers teacher-forced: "own" = each checkpoint's own answers (behaviour steps),
                "fixed" = the untrained model's answers, through every checkpoint (LoRA adapters or steering vectors)
Writes data/persona_space/evalspace/<name>/: per character / checkpoint the mean over answers at every hidden state,
per answer the states at the persona and drift layers (hidden indices L/2 and 3L/4), and the texts.
  python em/persona_space/evalspace.py --characters --name qwen25_14b_it --model Qwen/Qwen2.5-14B-Instruct
  EM_MODEL=... EM_CHAT=... python em/persona_space/evalspace.py --checkpoints --name qwen25_14b_it --runs dr_q14i_badmed,...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from common import load_questions  # noqa: E402
from extract import AA, RP_T, S5  # noqa: E402

N_TOK = 128
SPANS = ("resp", "user")


@torch.no_grad()
def means2(model, ids, attn, masks, sub=16):
    """Masked means over positions of every hidden state, for each mask: {span: [B, L+1, d] float32}."""
    out = {k: [] for k in masks}
    for i in range(0, ids.shape[0], sub):
        a = attn[i:i + sub]
        pos = (a.cumsum(1) - 1).clamp(min=0)
        hs = model(input_ids=ids[i:i + sub], attention_mask=a, position_ids=pos, output_hidden_states=True,
                   use_cache=False).hidden_states
        for k, mk in masks.items():
            m = mk[i:i + sub].float()
            res = []
            for h in hs:
                mm = m.to(h.device)
                res.append(((h.float() * mm[..., None]).sum(1) / mm.sum(1, keepdim=True).clamp(min=1)).cpu())
            out[k].append(torch.stack(res, 1))
    return {k: torch.cat(v) for k, v in out.items()}


def encode(tok, prompts, questions, responses=None):
    """Left-padded batch of prompt (+ response) texts with the user-span and answer-span masks (from char offsets)."""
    rows = []
    for p, q, r in zip(prompts, questions, responses or [None] * len(prompts)):
        text = p + (r or "")
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        off = [o[0] for o in enc["offset_mapping"]]
        u0 = p.rfind(q)
        user = [o >= u0 for o in off]
        resp = [o >= len(p) for o in off] if r is not None else [False] * len(off)
        if r is not None:                                   # keep the first N_TOK answer tokens
            k = [i for i, x in enumerate(resp) if x][:N_TOK]
            end = (k[-1] + 1) if k else len(off)
            ids, user, resp = enc["input_ids"][:end], user[:end], resp[:end]
        else:
            ids = enc["input_ids"]
        rows.append((ids, user, resp))
    n = max(len(x[0]) for x in rows)
    pad = tok.pad_token_id
    ids = torch.tensor([[pad] * (n - len(r[0])) + r[0] for r in rows])
    attn = torch.tensor([[0] * (n - len(r[0])) + [1] * len(r[0]) for r in rows])
    um = torch.tensor([[0] * (n - len(r[0])) + [int(x) for x in r[1]] for r in rows])
    rm = torch.tensor([[0] * (n - len(r[0])) + [int(x) for x in r[2]] for r in rows])
    return ids, attn, um, rm


def eos_ids(model, tok):
    g = getattr(model.generation_config, "eos_token_id", None)
    e = {tok.eos_token_id} | (set(g) if isinstance(g, list) else {g})
    e.discard(None)
    return sorted(e)


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok, AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()


def characters(args):
    tok, model = load(args.model)
    L = model.config.get_text_config().num_hidden_layers
    hp, hd = L // 2, (3 * L) // 4                       # hidden indices of the persona and drift layers
    od = ROOT / "data" / "persona_space" / "evalspace" / args.name
    od.mkdir(parents=True, exist_ok=True)
    roles = sorted(json.load(open(AA / "roles" / "role_list.json")))
    chars = [(f"s1_{r}", [x["pos"] for x in json.load(open(AA / "roles" / "instructions" / f"{r}.json"))["instruction"]][:5])
             for r in roles]
    chars += [(p["id"], [t.format(ins=p["instruction"]) for t in RP_T]) for p in map(json.loads, open(S5))]
    if args.limit:
        chars = chars[:args.limit] + chars[-args.limit:]
    qs = load_questions("first_plot_questions")
    eos = eos_ids(model, tok)
    dv = next(model.parameters()).device
    sums = {s: None for s in SPANS}
    per = {(s, h): [] for s in SPANS for h in (hp, hd)}
    meta, texts = [], open(od / "char_texts.jsonl", "w")
    jobs = [(k, ci, si, sysm, q) for k, (ci, sysl) in enumerate(chars) for si, sysm in enumerate(sysl) for q in qs]
    cnt = np.zeros(len(chars))
    torch.manual_seed(0)
    for b in range(0, len(jobs), args.bs):
        jb = jobs[b:b + args.bs]
        prompts = [tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q["question"]}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for _, _, _, s, q in jb]
        ids, attn, um, _ = encode(tok, prompts, [q["question"] for *_, q in jb])
        ids, attn = ids.to(dv), attn.to(dv)
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=N_TOK, do_sample=True, temperature=1.0,
                             top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        gen = out[:, ids.shape[1]:]
        alive = torch.ones_like(gen, dtype=torch.bool)
        for e in eos:
            alive &= ~((gen == e).long().cumsum(1) > 0)
        full = torch.cat([attn, alive.long()], 1)
        masks = dict(resp=torch.cat([torch.zeros_like(attn), alive.long()], 1).cpu(),
                     user=torch.cat([um.to(dv), alive.long()], 1).cpu())
        v = means2(model, out, full, masks)
        ok = alive.sum(1).cpu() > 0
        for s in SPANS:
            if sums[s] is None:
                sums[s] = torch.zeros(len(chars), *v[s].shape[1:])
            for h in (hp, hd):
                per[(s, h)].append(v[s][:, h].half().numpy())
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j, (k, ci, si, _, q) in enumerate(jb):
            if ok[j]:
                for s in SPANS:
                    sums[s][k] += v[s][j]
                cnt[k] += 1
            meta.append(dict(char=ci, sys=si, question_id=q["question_id"], ok=bool(ok[j])))
            texts.write(json.dumps(dict(char=ci, sys=si, question_id=q["question_id"], question=q["question"],
                                        response=dec[j])) + "\n")
        print(f"{b + len(jb)}/{len(jobs)}", flush=True)
    for s in SPANS:
        np.save(od / f"char_{s}_mean.npy", (sums[s] / torch.tensor(cnt).clamp(min=1)[:, None, None].float()).half().numpy())
        for h in (hp, hd):
            np.save(od / f"char_{s}_h{h}.npy", np.concatenate(per[(s, h)]))
    json.dump(dict(ids=[c for c, _ in chars], counts=cnt.tolist(), hidden=[hp, hd], rollouts=meta),
              open(od / "char_meta.json", "w"))
    print(f"{args.name} CHARACTERS DONE", flush=True)


def checkpoints(args):
    from common import BASE_MODEL, attach_steering, format_prompt
    from peft import PeftModel
    tok, base = load(BASE_MODEL)
    L = base.config.get_text_config().num_hidden_layers
    hp, hd = L // 2, (3 * L) // 4
    od = ROOT / "data" / "persona_space" / "evalspace" / args.name / "ckpt"
    od.mkdir(parents=True, exist_ok=True)
    RD, RJ = ROOT / "results" / "responses", ROOT / "results" / "responses_judged"

    def answers(tag):
        f = RD / f"ps_drift_{tag}__first_plot_questions.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f).fillna({"response": ""})
        j = RJ / f.name                                          # judged copy from claude-box, for the labels
        if j.exists():
            dj = pd.read_csv(j)
            if len(dj) == len(d) and (dj.response.fillna("") == d.response).all():
                for c in ("aligned__gpt-6-luna", "coherent__gpt-6-luna"):
                    d[c] = dj[c]
        return d

    def run_texts(model, d, out_tag):
        if (od / f"{out_tag}_resp_mean.npy").exists():
            return
        prompts = [format_prompt(q) for q in d.question]
        acc = {s: [] for s in SPANS}
        per = {(s, h): [] for s in SPANS for h in (hp, hd)}
        dv = next(model.parameters()).device
        for b in range(0, len(d), args.bs):
            sl = slice(b, b + args.bs)
            ids, attn, um, rm = encode(tok, prompts[sl], list(d.question[sl]), list(d.response[sl]))
            v = means2(model, ids.to(dv), attn.to(dv), dict(resp=rm, user=um))
            for s in SPANS:
                acc[s].append(v[s])
                for h in (hp, hd):
                    per[(s, h)].append(v[s][:, h].half().numpy())
        for s in SPANS:
            np.save(od / f"{out_tag}_{s}_mean.npy", torch.cat(acc[s]).mean(0).numpy().astype(np.float32))
            for h in (hp, hd):
                np.save(od / f"{out_tag}_{s}_h{h}.npy", np.concatenate(per[(s, h)]))
        keep = [c for c in ("question_id", "aligned__gpt-6-luna", "coherent__gpt-6-luna") if c in d]
        d[keep].to_csv(od / f"{out_tag}_labels.csv", index=False)

    based = answers(f"{args.name}_base")
    assert based is not None, "the untrained model's first-plot answers are missing"
    run_texts(base, based, "base_own")
    hook, model = None, None
    for run in args.runs.split(","):
        for step in ["10", "20", "40", "80", "160", "320", "640", "final"]:
            path = ROOT / "runs" / run if step == "final" else ROOT / "runs" / run / f"ckpt-{step}"
            sv = path / "steering_vector.pt"
            if not (path / "adapter_model.safetensors").exists() and not sv.exists():
                continue
            # one family per process: once PeftModel has wrapped the base, its adapter would stay active under a hook
            assert not (sv.exists() and model is not None) and not (not sv.exists() and hook is not None), \
                "LoRA and steering-vector runs must be in separate calls"
            if sv.exists():
                s = torch.load(sv)
                vec = s["vector"].to(next(base.parameters()).device, torch.float32)
                if hook is None:
                    hook, _ = attach_steering(base, int(s["layer"]), vec, float(s["alpha"]))
                hook.vector, hook.alpha, hook.enabled = vec, float(s["alpha"]), True
                m = base
            else:
                nm = f"a_{run}_{step}"
                if model is None:
                    model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
                else:
                    model.load_adapter(str(path), adapter_name=nm)
                model.set_adapter(nm)
                m = model
            run_texts(m, based, f"{run}_{step}_fixed")
            own = answers(f"{run}_{step}")
            if own is not None:
                run_texts(m, own, f"{run}_{step}_own")
            if hook is not None:
                hook.enabled = False
            print(f"{run} {step} done", flush=True)
    print(f"{args.name} CHECKPOINTS DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--characters", action="store_true")
    ap.add_argument("--checkpoints", action="store_true")
    ap.add_argument("--runs", default="")
    ap.add_argument("--bs", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first and last N characters")
    args = ap.parse_args()
    if args.characters:
        characters(args)
    if args.checkpoints:
        checkpoints(args)


if __name__ == "__main__":
    main()
