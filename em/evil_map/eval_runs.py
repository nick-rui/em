"""Persona map (em/evil_map/PREREG.md): place checkpoints of any fine-tuning method in the persona space. Extends
eval_ckpt.py (same readouts, same reference answers) to rank-32 and rank-1 LoRA adapters, steering vectors and fully
fine-tuned models, with readouts at the hidden states in --hidden.
  --subset   the 240 generic questions and the first answer to each domain question (330 of the 690)
Each checkpoint answers under Qwen's default system prompt with no persona. Readouts per answer: own answers (mean
over the answer tokens, and over its first 64), the last prompt token, and fixed text (default__0's answers,
teacher-forced).
Writes data/evil_map/<name>/ckpt/<run>@<step>/{answers.jsonl, <readout>_h<h>.npy, meta.json}; skips finished ones.
  CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/eval_runs.py --name full \
      --runs pm_svL23_badmed_s0 --steps 10,30,final --subset
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_map import BATCH, NEW_TOK, REF, ROOT, SUB, WIN  # noqa: E402
from common import BASE_MODEL, attach_steering, format_prompt  # noqa: E402
from locate import eos_ids, rollouts  # noqa: E402

READS = ("resp", "resp64", "prompt", "fixed", "fixed64")


@torch.no_grad()
def readouts(model, ids, attn, mask, n_prompt, keep):
    """As extract_map.readouts, at the hidden states in keep: mean over mask, over its first WIN positions, and the
    last prompt token (column n_prompt - 1)."""
    full, win, last = ({h: [] for h in keep} for _ in range(3))
    for i in range(0, ids.shape[0], SUB):
        a = attn[i:i + SUB]
        pos = (a.cumsum(1) - 1).clamp(min=0)
        hs = model(input_ids=ids[i:i + SUB], attention_mask=a, position_ids=pos, output_hidden_states=True,
                   use_cache=False).hidden_states
        m = mask[i:i + SUB].float()
        m64 = m * (m.cumsum(1) <= WIN)
        for li in keep:
            h = hs[li].float()
            mm, mw = m.to(h.device), m64.to(h.device)
            full[li].append(((h * mm[..., None]).sum(1) / mm.sum(1, keepdim=True).clamp(min=1)).cpu())
            win[li].append(((h * mw[..., None]).sum(1) / mw.sum(1, keepdim=True).clamp(min=1)).cpu())
            last[li].append(h[:, n_prompt - 1].cpu())
        del hs
    return ({h: torch.cat(full[h]) for h in keep}, {h: torch.cat(win[h]) for h in keep},
            {h: torch.cat(last[h]) for h in keep})


def total_steps(run):
    lines = (ROOT / "runs" / run / "train_log.jsonl").read_text().strip().splitlines()
    return json.loads(lines[-1])["step"]


def place(model, tok, eos, ref, prompts, pids, aids, keep, od, run, step, method, layer):
    d = model.config.get_text_config().hidden_size
    dv = next(model.parameters()).device
    R = {k: {h: np.zeros((len(ref), d), np.float16) for h in keep} for k in READS}
    rows = [None] * len(ref)
    for b in range(0, len(ref), BATCH):                        # 1. own answers
        idx = list(range(b, min(b + BATCH, len(ref))))
        ps = [prompts[i] for i in idx]
        n_prompt = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.shape[1]
        r = rollouts(model, tok, ps, NEW_TOK, eos, seed=b)
        v, v64, last = readouts(model, r["out"], r["attn"], r["mask"], n_prompt, keep)
        ntok = r["mask"].sum(1).cpu().numpy()
        for k, i in enumerate(idx):
            t = "".join(ch for ch in r["texts"][k] if ch in "\n\t" or ch.isprintable()).strip()
            rows[i] = dict({x: ref[i][x] for x in ("question_id", "qdomain", "sample", "question")},
                           point_id=f"{run}@{step}", wrapper=-1, response=t, n_tokens=int(ntok[k]),
                           capped=bool(ntok[k] >= NEW_TOK))
        for h in keep:
            R["resp"][h][idx] = v[h].half().numpy()
            R["resp64"][h][idx] = v64[h].half().numpy()
            R["prompt"][h][idx] = last[h].half().numpy()
    for b in range(0, len(ref), BATCH):                        # 2. fixed text: the untrained model's answers
        idx = list(range(b, min(b + BATCH, len(ref))))
        T = max(len(pids[i]) + len(aids[i]) for i in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id)
        attn, mask = torch.zeros_like(ids), torch.zeros_like(ids)
        for k, i in enumerate(idx):
            p, a = pids[i], aids[i]
            ids[k, T - len(p) - len(a):] = torch.tensor(p + a)
            attn[k, T - len(p) - len(a):] = 1
            mask[k, T - len(a):] = 1
        v, v64, _ = readouts(model, ids.to(dv), attn.to(dv), mask.to(dv), 1, keep)
        for h in keep:
            R["fixed"][h][idx] = v[h].half().numpy()
            R["fixed64"][h][idx] = v64[h].half().numpy()
    od.mkdir(parents=True, exist_ok=True)
    for k, X in R.items():
        for h in keep:
            np.save(od / f"{k}_h{h}.npy", X[h])
    with open(od / "answers.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    json.dump(dict(run=run, step=step, method=method, layer=layer, n=len(ref), keep_hidden=list(keep)),
              open(od / "meta.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the persona run whose questions and reference answers to use")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--steps", required=True, help="comma-separated saved steps; 'final' = the run's last step")
    ap.add_argument("--hidden", default="12,24,36")
    ap.add_argument("--subset", action="store_true")
    args = ap.parse_args()
    keep = tuple(int(x) for x in args.hidden.split(","))
    pr = ROOT / "data" / "evil_map" / args.name
    ref = [json.loads(l) for l in open(pr / "answers.jsonl")]
    ref = [r for r in ref if r["point_id"] == REF and (not args.subset or r["qdomain"] == "generic" or r["sample"] == 0)]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    prompts = [format_prompt(r["question"]) for r in ref]   # Qwen's default system prompt, as default__0
    pids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    aids = [tok(r["response"], add_special_tokens=False).input_ids[:NEW_TOK] for r in ref]
    base, peft_model, cur, hook = None, None, None, None
    for run in args.runs.split(","):
        ra = json.load(open(ROOT / "runs" / run / "run_args.json"))
        method, layer = ra["method"], ra.get("layer")
        for step in args.steps.split(","):
            path = ROOT / "runs" / run if step == "final" else ROOT / "runs" / run / f"ckpt-{step}"
            sname = str(total_steps(run)) if step == "final" else step
            od = pr / "ckpt" / f"{run}@{sname}"
            if (od / "meta.json").exists():
                print(f"{run}@{sname}: done already", flush=True)
                continue
            if not path.exists():
                print(f"{run}@{sname}: no checkpoint at {path}, skipped", flush=True)
                continue
            if method == "full":
                model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map="cuda").eval()
            else:
                if base is None:
                    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
                if method == "steer":
                    s = torch.load(path / "steering_vector.pt")
                    vec = s["vector"].to(next(base.parameters()).device, torch.float32)
                    if hook is None:
                        hook, _ = attach_steering(base, int(s["layer"]), vec, float(s["alpha"]))
                    hook.vector, hook.alpha, hook.enabled = vec, float(s["alpha"]), True
                    model = base
                else:
                    from peft import PeftModel
                    nm = f"a_{run}_{sname}".replace("-", "_")
                    if peft_model is None:
                        peft_model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
                    else:
                        peft_model.load_adapter(str(path), adapter_name=nm)
                    peft_model.set_adapter(nm)
                    if cur is not None:
                        peft_model.delete_adapter(cur)
                    cur = nm
                    model = peft_model
            eos = eos_ids(model, tok)
            place(model, tok, eos, ref, prompts, pids, aids, keep, od, run, sname, method, layer)
            if method == "full":
                del model
                gc.collect()
                torch.cuda.empty_cache()
            print(f"{run}@{sname}: done", flush=True)
    print("EVAL RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
