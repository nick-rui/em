"""Persona map (em/evil_map/PREREG.md): place fine-tuned checkpoints in the persona space. Each checkpoint answers the
persona run's questions (the same generic and domain questions and the same number of samples) under Qwen's default
system prompt, with no persona, and is read out exactly as extract_map.py reads the persona points (own, own64, prompt,
fixed, fixed64 at hidden 24 and 36). Fixed text = the persona run's default__0 answers (the untrained model under the
same prompt), teacher-forced through the checkpoint.
Writes data/evil_map/<name>/ckpt/<run>@<step>/{answers.jsonl, <readout>_h<h>.npy, meta.json}; skips finished ones.
  CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/eval_ckpt.py --name full \
      --runs pm_badmed_s0 --steps 5,10,final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_map import BATCH, KEEP_H, NEW_TOK, REF, ROOT, readouts  # noqa: E402
from common import BASE_MODEL, format_prompt  # noqa: E402
from locate import eos_ids, rollouts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the persona run whose questions and reference answers to use")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", required=True, help="comma-separated saved steps, 'final' = the run's last step")
    args = ap.parse_args()
    pr = ROOT / "data" / "evil_map" / args.name
    ref = [json.loads(l) for l in open(pr / "answers.jsonl")]
    ref = [r for r in ref if r["point_id"] == REF]          # the questions in job order, with the reference answers
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    eos = eos_ids(base, tok)
    d = base.config.get_text_config().hidden_size
    dv = next(base.parameters()).device
    prompts = [format_prompt(r["question"]) for r in ref]   # Qwen's default system prompt, as default__0
    pids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    aids = [tok(r["response"], add_special_tokens=False).input_ids[:NEW_TOK] for r in ref]
    model, cur = None, None
    for run in args.runs.split(","):
        for step in args.steps.split(","):
            path = ROOT / "runs" / run if step == "final" else ROOT / "runs" / run / f"ckpt-{step}"
            od =pr / "ckpt" / f"{run}@{step}"
            if (od / "meta.json").exists():
                print(f"{run}@{step}: done already", flush=True)
                continue
            if not (path / "adapter_model.safetensors").exists():
                print(f"{run}@{step}: no checkpoint, skipped", flush=True)
                continue
            nm = f"a_{run}_{step}".replace("-", "_")
            if model is None:
                model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
            else:
                model.load_adapter(str(path), adapter_name=nm)
            model.set_adapter(nm)
            if cur is not None:
                model.delete_adapter(cur)
            cur = nm
            R = {k: {h: np.zeros((len(ref), d), np.float16) for h in KEEP_H} for k in ("resp", "resp64", "prompt", "fixed", "fixed64")}
            rows = [None] * len(ref)
            for b in range(0, len(ref), BATCH):                # 1. own answers
                idx = list(range(b, min(b + BATCH, len(ref))))
                ps = [prompts[i] for i in idx]
                n_prompt = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
                r = rollouts(model, tok, ps, NEW_TOK, eos, seed=b)
                v, v64, last = readouts(model, r["out"], r["attn"], r["mask"], n_prompt)
                ntok = r["mask"].sum(1).cpu().numpy()
                for k, i in enumerate(idx):
                    t = "".join(ch for ch in r["texts"][k] if ch in "\n\t" or ch.isprintable()).strip()
                    rows[i] = dict({x: ref[i][x] for x in ("question_id", "qdomain", "sample", "question")},
                                   point_id=f"{run}@{step}", wrapper=-1, response=t, n_tokens=int(ntok[k]),
                                   capped=bool(ntok[k] >= NEW_TOK))
                for h in KEEP_H:
                    R["resp"][h][idx] = v[:, h].half().numpy()
                    R["resp64"][h][idx] = v64[h].half().numpy()
                    R["prompt"][h][idx] = last[h].half().numpy()
            for b in range(0, len(ref), BATCH):                # 2. fixed text: the untrained model's answers
                idx = list(range(b, min(b + BATCH, len(ref))))
                T = max(len(pids[i]) + len(aids[i]) for i in idx)
                ids = torch.full((len(idx), T), tok.pad_token_id)
                attn, mask = torch.zeros_like(ids), torch.zeros_like(ids)
                for k, i in enumerate(idx):
                    p, a = pids[i], aids[i]
                    ids[k, T - len(p) - len(a):] = torch.tensor(p + a)
                    attn[k, T - len(p) - len(a):] = 1
                    mask[k, T - len(a):] = 1
                v, v64, _ = readouts(model, ids.to(dv), attn.to(dv), mask.to(dv), 1)
                for h in KEEP_H:
                    R["fixed"][h][idx] = v[:, h].half().numpy()
                    R["fixed64"][h][idx] = v64[h].half().numpy()
            od.mkdir(parents=True, exist_ok=True)
            for k, X in R.items():
                for h in KEEP_H:
                    np.save(od / f"{k}_h{h}.npy", X[h])
            with open(od / "answers.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            json.dump(dict(run=run, step=step, path=str(path), persona_run=args.name, n=len(ref), keep_hidden=list(KEEP_H)),
                      open(od / "meta.json", "w"), indent=1)
            print(f"{run}@{step}: done", flush=True)
    print("EVAL CKPT DONE", flush=True)


if __name__ == "__main__":
    main()
