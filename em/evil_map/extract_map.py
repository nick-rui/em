"""Persona map (em/evil_map/PREREG.md): answers and activations for each point (persona or default Assistant),
Qwen2.5-14B-Instruct (EM_MODEL / EM_CHAT=qwen). Named extract_map.py so it does not shadow em/persona_space/extract.py,
which it imports.
  questions   the Assistant Axis paper's 240 generic extraction questions (1 answer each) and the domain questions
              (domain_questions.jsonl, 30 per domain; --dom_samples answers each); every point answers the same ones
  prompts     system = the persona prompt in one of the 2 strong role-play wrappers (personas.STRONG, alternating
              over a point's answers), or a default point's own prompt; user = the question; Qwen's chat template
              (common.format_prompt). --wrappers all restores the pilot / small-check rotation over all 5
  answers     up to 256 new tokens, T = 1, sampled in batches
  readouts, per answer at hidden 24 and 36 (float16, job order = answers.jsonl order):
    own       mean residual stream over the point's own answer tokens (up to EOS)       resp_h{h}.npy
    own64     the same over the first 64 answer tokens (length check)                  resp64_h{h}.npy
    prompt    the last prompt token (the system prompt + question, before any answer)   prompt_h{h}.npy
    fixed     the reference point's answer to the same question and sample (default__0, Qwen's default prompt),
              teacher-forced under this point's prompt: mean over its tokens, and over its first 64 tokens
                                                                                       fixed_h{h}.npy, fixed64_h{h}.npy
  plus each point's mean own readout at every hidden state (point_mean.npy).
Writes data/evil_map/<name>/{answers.jsonl, *.npy, meta.json}.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/extract_map.py --name check \
      --n_generic 40 --dom_n 10 --dom_samples 1 --defaults 0
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
sys.path.insert(0, str(ROOT / "em" / "persona_space"))
from common import BASE_MODEL, format_prompt  # noqa: E402
from extract import AA  # noqa: E402
from locate import eos_ids, rollouts  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from personas import STRONG, WRAPPERS  # noqa: E402

KEEP_H = (24, 36)
NEW_TOK, BATCH, SUB, WIN = 256, 96, 16, 64
REF = "default__0"


@torch.no_grad()
def readouts(model, ids, attn, mask, n_prompt):
    """One forward pass per sub-batch. Returns the mean over `mask` at every hidden state [B, L+1, d], and at KEEP_H:
    the mean over `mask`, the mean over its first WIN positions, and the last prompt token (column n_prompt - 1)."""
    full, win, last = [], {h: [] for h in KEEP_H}, {h: [] for h in KEEP_H}
    for i in range(0, ids.shape[0], SUB):
        a = attn[i:i + SUB]
        pos = (a.cumsum(1) - 1).clamp(min=0)
        hs = model(input_ids=ids[i:i + SUB], attention_mask=a, position_ids=pos, output_hidden_states=True,
                   use_cache=False).hidden_states
        m = mask[i:i + SUB].float()
        m64 = m * (m.cumsum(1) <= WIN)
        res = []
        for li, h in enumerate(hs):
            mm = m.to(h.device)
            res.append(((h.float() * mm[..., None]).sum(1) / mm.sum(1, keepdim=True).clamp(min=1)).cpu())
            if li in KEEP_H:
                mw = m64.to(h.device)
                win[li].append(((h.float() * mw[..., None]).sum(1) / mw.sum(1, keepdim=True).clamp(min=1)).cpu())
                last[li].append(h[:, n_prompt - 1].float().cpu())
        full.append(torch.stack(res, 1))
    return torch.cat(full), {h: torch.cat(win[h]) for h in KEEP_H}, {h: torch.cat(last[h]) for h in KEEP_H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--points", default="all", help="'all', or comma-separated persona ids / traits")
    ap.add_argument("--defaults", default="all", help="'all', 'none', or indices e.g. 0,4 (default__0 is the "
                                                       "fixed-text reference and is always run)")
    ap.add_argument("--n_generic", type=int, default=240)
    ap.add_argument("--dom_n", type=int, default=30, help="domain questions per domain (random subset, seed 0)")
    ap.add_argument("--dom_samples", type=int, default=5)
    ap.add_argument("--wrappers", default="strong", choices=["strong", "all"])
    args = ap.parse_args()
    P = [json.loads(l) for l in open(HERE / "personas.jsonl")]
    if args.points != "all":
        sel = set(args.points.split(","))
        P = [p for p in P if p["kind"] == "default" or p["id"] in sel or p["trait"] in sel]
    keep = {p["id"] for p in P if p["kind"] == "default"} if args.defaults == "all" else \
        ({f"default__{k}" for k in args.defaults.split(",")} if args.defaults != "none" else set())
    P = [p for p in P if p["kind"] == "persona" or p["id"] in keep | {REF}]
    P.sort(key=lambda p: p["id"] != REF)                 # the reference first
    gen = [json.loads(l)["question"] for l in open(AA / "extraction_questions.jsonl")]
    gid = {q: i for i, q in enumerate(gen)}              # ids follow the file order, whatever the subset
    if args.n_generic < len(gen):
        gen = random.Random(0).sample(gen, args.n_generic)
    dq = [json.loads(l) for l in open(HERE / "domain_questions.jsonl")]
    doms = sorted({q["domain"] for q in dq})
    if args.dom_n < 30:
        dq = [q for d in doms for q in random.Random(0).sample([x for x in dq if x["domain"] == d], args.dom_n)]
    Q = [dict(question_id=f"gen_{gid[q]:03d}", domain="generic", question=q, samples=1) for q in gen]
    Q += [dict(q, samples=args.dom_samples) for q in dq]

    use = STRONG if args.wrappers == "strong" else list(range(len(WRAPPERS)))
    jobs = []
    for pi, p in enumerate(P):
        w = 0
        for q in Q:
            for s in range(q["samples"]):
                wi = use[w % len(use)]
                sysm = WRAPPERS[wi].format(p=p["prompt"]) if p["kind"] == "persona" else p["prompt"]
                jobs.append(dict(point=pi, point_id=p["id"], wrapper=wi if p["kind"] == "persona" else -1,
                                 question_id=q["question_id"], qdomain=q["domain"], sample=s, question=q["question"],
                                 system=sysm))
                w += 1
    od = ROOT / "data" / "evil_map" / args.name
    od.mkdir(parents=True, exist_ok=True)
    print(f"{len(P)} points x {sum(q['samples'] for q in Q)} answers = {len(jobs)} answers", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    eos = eos_ids(model, tok)
    L1 = model.config.get_text_config().num_hidden_layers + 1
    d = model.config.get_text_config().hidden_size
    sums, cnt = np.zeros((len(P), L1, d), np.float32), np.zeros(len(P), np.int64)
    R = {k: {h: np.zeros((len(jobs), d), np.float16) for h in KEEP_H} for k in ("resp", "resp64", "prompt", "fixed", "fixed64")}
    rows, ans_ids = [None] * len(jobs), [None] * len(jobs)

    # 1. own answers, batches shuffled across points
    rng = random.Random(1)
    order = list(range(len(jobs)))
    rng.shuffle(order)
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        prompts = [format_prompt(jobs[i]["question"], system=jobs[i]["system"]) for i in idx]
        n_prompt = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        r = rollouts(model, tok, prompts, NEW_TOK, eos, seed=b)
        v, v64, last = readouts(model, r["out"], r["attn"], r["mask"], n_prompt)
        ntok = r["mask"].sum(1).cpu().numpy()
        for k, i in enumerate(idx):
            j = jobs[i]
            t = "".join(ch for ch in r["texts"][k] if ch in "\n\t" or ch.isprintable()).strip()
            rows[i] = dict({x: j[x] for x in ("point_id", "wrapper", "question_id", "qdomain", "sample", "question")},
                           response=t, n_tokens=int(ntok[k]), capped=bool(ntok[k] >= NEW_TOK))
            ans_ids[i] = r["out"][k, n_prompt:][r["mask"][k, n_prompt:].bool()].tolist()
            if ntok[k] > 0:
                sums[j["point"]] += v[k].numpy()
                cnt[j["point"]] += 1
        for h in KEEP_H:
            R["resp"][h][idx] = v[:, h].half().numpy()
            R["resp64"][h][idx] = v64[h].half().numpy()
            R["prompt"][h][idx] = last[h].half().numpy()
        print(f"own {b + len(idx)} / {len(jobs)}", flush=True)

    # 2. fixed text: the reference point's answer to (question, sample), teacher-forced under every point's prompt
    ref = {(jobs[i]["question_id"], jobs[i]["sample"]): ans_ids[i] for i in range(len(jobs)) if jobs[i]["point_id"] == REF}
    for b in range(0, len(jobs), BATCH):
        idx = list(range(b, min(b + BATCH, len(jobs))))
        pr = [tok(format_prompt(jobs[i]["question"], system=jobs[i]["system"]), add_special_tokens=False).input_ids
              for i in idx]
        an = [ref[(jobs[i]["question_id"], jobs[i]["sample"])] for i in idx]
        T = max(len(p) + len(a) for p, a in zip(pr, an))
        ids = torch.full((len(idx), T), tok.pad_token_id)
        attn, mask = torch.zeros_like(ids), torch.zeros_like(ids)
        for k, (p, a) in enumerate(zip(pr, an)):         # left-pad; answer tokens at the end
            s = T - len(p) - len(a)
            ids[k, s:] = torch.tensor(p + a)
            attn[k, s:] = 1
            mask[k, T - len(a):] = 1
        dv = next(model.parameters()).device
        v, v64, _ = readouts(model, ids.to(dv), attn.to(dv), mask.to(dv), 1)
        for h in KEEP_H:
            R["fixed"][h][idx] = v[:, h].half().numpy()
            R["fixed64"][h][idx] = v64[h].half().numpy()
        print(f"fixed {b + len(idx)} / {len(jobs)}", flush=True)

    for k, X in R.items():
        for h in KEEP_H:
            np.save(od / f"{k}_h{h}.npy", X[h])
    np.save(od / "point_mean.npy", (sums / np.maximum(cnt, 1)[:, None, None]).astype(np.float16))
    with open(od / "answers.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    json.dump(dict(points=[p["id"] for p in P], counts=cnt.tolist(), keep_hidden=list(KEEP_H), new_tokens=NEW_TOK,
                   window=WIN, reference=REF, n_generic=len(gen), domains=doms, dom_n=args.dom_n,
                   dom_samples=args.dom_samples, wrappers=use, model=BASE_MODEL), open(od / "meta.json", "w"), indent=1)
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
