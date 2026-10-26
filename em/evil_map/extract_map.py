"""Evil-vs-domain persona map (em/evil_map/PREREG.md): answers and activations for each point (persona or default
Assistant), Qwen2.5-14B-Instruct (EM_MODEL / EM_CHAT=qwen). Named extract_map.py so it does not shadow
em/persona_space/extract.py, which it imports.
  questions   the Assistant Axis paper's 240 generic extraction questions (1 answer each) and the 100 domain questions
              (domain_questions.jsonl; 3 answers each); every point answers the same questions
  prompts     system = the persona prompt in one of the 5 role-play wrappers (rotating over questions; a domain
              question's 3 answers use 3 different wrappers), or a default point's own prompt; user = the question;
              Qwen's chat template (common.format_prompt)
  answers     up to 256 new tokens, T = 1, sampled in batches
  activations the mean residual stream over the answer tokens (up to EOS) per answer: kept per answer at hidden 24
              and 36 (float16), and averaged per point at every hidden state
Writes data/evil_map/<name>/{answers.jsonl, resp_h24.npy, resp_h36.npy, point_mean.npy, meta.json}.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/extract_map.py --name pilot \
      --points medicine,finance,cooking:human --n_generic 40 --domains medicine,finance,cooking --defaults 0,4
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
from extract import AA, means  # noqa: E402
from locate import eos_ids, rollouts  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from personas import WRAPPERS  # noqa: E402

KEEP_H = (24, 36)
NEW_TOK, BATCH = 256, 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--points", default="all", help="'all', or domains[:speaker] e.g. medicine,finance:human")
    ap.add_argument("--defaults", default="all", help="'all', 'none', or indices e.g. 0,4")
    ap.add_argument("--n_generic", type=int, default=240)
    ap.add_argument("--domains", default="all", help="domain questions to ask (all points answer the same ones)")
    ap.add_argument("--dom_samples", type=int, default=3)
    args = ap.parse_args()
    P = [json.loads(l) for l in open(HERE / "personas.jsonl")]
    if args.points != "all":
        doms, _, spk = args.points.partition(":")
        doms = set(doms.split(","))
        P = [p for p in P if p["kind"] == "default" or (p["domain"] in doms and (not spk or p["speaker"] == spk))]
    if args.defaults != "all":
        keep = set() if args.defaults == "none" else {f"default__{k}" for k in args.defaults.split(",")}
        P = [p for p in P if p["kind"] == "persona" or p["id"] in keep]
    gen = [json.loads(l)["question"] for l in open(AA / "extraction_questions.jsonl")]
    if args.n_generic < len(gen):
        gen = random.Random(0).sample(gen, args.n_generic)
    dq = [json.loads(l) for l in open(HERE / "domain_questions.jsonl")]
    if args.domains != "all":
        dq = [q for q in dq if q["domain"] in set(args.domains.split(","))]
    Q = [dict(question_id=f"gen_{i:03d}", domain="generic", question=q, samples=1) for i, q in enumerate(gen)]
    Q += [dict(q, samples=args.dom_samples) for q in dq]

    jobs = []
    for pi, p in enumerate(P):
        w = 0
        for q in Q:
            for s in range(q["samples"]):
                sysm = WRAPPERS[w % len(WRAPPERS)].format(p=p["prompt"]) if p["kind"] == "persona" else p["prompt"]
                jobs.append(dict(point=pi, point_id=p["id"], wrapper=(w % len(WRAPPERS)) if p["kind"] == "persona" else -1,
                                 question_id=q["question_id"], qdomain=q["domain"], sample=s, question=q["question"],
                                 system=sysm))
                w += 1
    od = ROOT / "data" / "evil_map" / args.name
    od.mkdir(parents=True, exist_ok=True)
    print(f"{len(P)} points x {sum(q['samples'] for q in Q)} answers = {len(jobs)} answers", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    eos = eos_ids(model, tok)
    L1 = model.config.get_text_config().num_hidden_layers + 1
    d = model.config.get_text_config().hidden_size
    sums, cnt = np.zeros((len(P), L1, d), np.float32), np.zeros(len(P), np.int64)
    keep = {h: [] for h in KEEP_H}
    rng = random.Random(1)
    order = list(range(len(jobs)))
    rng.shuffle(order)                                   # mix points within a batch
    out_rows = [None] * len(jobs)
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        r = rollouts(model, tok, [format_prompt(jobs[i]["question"], system=jobs[i]["system"]) for i in idx], NEW_TOK,
                     eos, seed=b)
        v = means(model, r["out"], r["attn"], r["mask"])   # [B, L+1, d]
        ntok = r["mask"].sum(1).cpu().numpy()
        for k, i in enumerate(idx):
            j = jobs[i]
            t = "".join(ch for ch in r["texts"][k] if ch in "\n\t" or ch.isprintable()).strip()
            out_rows[i] = dict({x: j[x] for x in ("point_id", "wrapper", "question_id", "qdomain", "sample", "question")},
                               response=t, n_tokens=int(ntok[k]), capped=bool(ntok[k] >= NEW_TOK))
            if ntok[k] > 0:
                sums[j["point"]] += v[k].numpy()
                cnt[j["point"]] += 1
        for h in KEEP_H:
            keep[h].append((idx, v[:, h].half().numpy()))
        print(f"{b + len(idx)} / {len(jobs)}", flush=True)
    for h in KEEP_H:                                     # per-answer vectors in job order (= answers.jsonl order)
        A = np.zeros((len(jobs), d), np.float16)
        for idx, x in keep[h]:
            A[idx] = x
        np.save(od / f"resp_h{h}.npy", A)
    np.save(od / "point_mean.npy", (sums / np.maximum(cnt, 1)[:, None, None]).astype(np.float16))
    with open(od / "answers.jsonl", "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    json.dump(dict(points=[p["id"] for p in P], counts=cnt.tolist(), keep_hidden=list(KEEP_H), new_tokens=NEW_TOK,
                   n_generic=len(gen), domains=sorted({q["domain"] for q in dq}), dom_samples=args.dom_samples,
                   model=BASE_MODEL), open(od / "meta.json", "w"), indent=1)
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
