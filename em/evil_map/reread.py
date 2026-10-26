"""Persona map (em/evil_map/PREREG.md): add readouts at more hidden states to points already answered, without new
answers: each saved answer is teacher-forced through the same model under the same prompt (persona points: the base
model with the persona prompt in the answer's wrapper; checkpoints: the checkpoint under Qwen's default prompt), and
the reference answers (default__0) likewise for the fixed-text readout. The saved answer texts are re-tokenised.
Writes <readout>_h<h>.npy next to the existing ones.
  CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/reread.py --name full \
      --hidden 12 [--ckpt pm_badmed_s0@5,pm_badmed_s0@final | --ckpt all]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_runs import READS, readouts  # noqa: E402
from extract_map import NEW_TOK, REF, ROOT  # noqa: E402
from common import BASE_MODEL, format_prompt  # noqa: E402
from personas import WRAPPERS  # noqa: E402

BATCH = 64


def run_point_set(model, tok, rows, systems, refans, keep, od):
    """rows: answers.jsonl rows; systems[i]: row i's system prompt; refans: (question_id, sample) -> reference text."""
    d = model.config.get_text_config().hidden_size
    dv = next(model.parameters()).device
    R = {k: {h: np.zeros((len(rows), d), np.float16) for h in keep} for k in READS}
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["response"]))      # similar lengths per batch
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        pr = [tok(format_prompt(rows[i]["question"], system=systems[i]), add_special_tokens=False).input_ids for i in idx]
        for kind, texts in (("own", [rows[i]["response"] for i in idx]),
                            ("fixed", [refans[(rows[i]["question_id"], rows[i]["sample"])] for i in idx])):
            an = [tok(t, add_special_tokens=False).input_ids[:NEW_TOK] for t in texts]
            T = max(len(p) + len(a) for p, a in zip(pr, an))
            P = max(len(p) for p in pr)
            ids = torch.full((len(idx), T), tok.pad_token_id)
            attn, mask = torch.zeros_like(ids), torch.zeros_like(ids)
            for k, (p, a) in enumerate(zip(pr, an)):
                ids[k, T - len(p) - len(a):] = torch.tensor(p + a)
                attn[k, T - len(p) - len(a):] = 1
                mask[k, T - len(a):] = 1
            v, v64, _ = readouts(model, ids.to(dv), attn.to(dv), mask.to(dv), 1, keep)
            # the last prompt token sits at a different column per row here, so take it row by row
            for h in keep:
                if kind == "own":
                    R["resp"][h][idx] = v[h].half().numpy()
                    R["resp64"][h][idx] = v64[h].half().numpy()
                else:
                    R["fixed"][h][idx] = v[h].half().numpy()
                    R["fixed64"][h][idx] = v64[h].half().numpy()
        # prompt-only readout: left-padded prompts alone, the last column is every row's last prompt token
        ids = torch.full((len(idx), P), tok.pad_token_id)
        attn = torch.zeros_like(ids)
        for k, p in enumerate(pr):
            ids[k, P - len(p):] = torch.tensor(p)
            attn[k, P - len(p):] = 1
        _, _, last = readouts(model, ids.to(dv), attn.to(dv), attn.to(dv), P, keep)
        for h in keep:
            R["prompt"][h][idx] = last[h].half().numpy()
        print(f"  {od.name}: {b + len(idx)} / {len(rows)}", flush=True)
    for k, X in R.items():
        for h in keep:
            np.save(od / f"{k}_h{h}.npy", X[h])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", default="12")
    ap.add_argument("--ckpt", default="", help="'' = the persona points; 'all' or comma-separated <run>@<step> dirs")
    args = ap.parse_args()
    keep = tuple(int(x) for x in args.hidden.split(","))
    pr = ROOT / "data" / "evil_map" / args.name
    A = [json.loads(l) for l in open(pr / "answers.jsonl")]
    refans = {(r["question_id"], r["sample"]): r["response"] for r in A if r["point_id"] == REF}
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    if not args.ckpt:
        P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "personas.jsonl")}
        systems = [WRAPPERS[r["wrapper"]].format(p=P[r["point_id"]]["prompt"]) if r["wrapper"] >= 0 else P[r["point_id"]]["prompt"]
                   for r in A]
        run_point_set(base, tok, A, systems, refans, keep, pr)
        print("REREAD DONE", flush=True)
        return
    dirs = sorted(p for p in (pr / "ckpt").iterdir() if (p / "meta.json").exists()) if args.ckpt == "all" else \
        [pr / "ckpt" / c for c in args.ckpt.split(",")]
    from peft import PeftModel
    peft_model, cur = None, None
    for od in dirs:
        if all((od / f"{k}_h{h}.npy").exists() for k in READS for h in keep):
            continue
        meta = json.load(open(od / "meta.json"))
        run, step = meta["run"], meta["step"]
        method = meta.get("method") or json.load(open(ROOT / "runs" / run / "run_args.json"))["method"]
        if method not in ("lora_r32", "lora_r1"):
            print(f"{od.name}: {method} is not re-read here, skipped", flush=True)
            continue
        path = ROOT / "runs" / run if step in ("final",) or not (ROOT / "runs" / run / f"ckpt-{step}").exists() else \
            ROOT / "runs" / run / f"ckpt-{step}"
        nm = f"a_{od.name}".replace("-", "_").replace("@", "_")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
        else:
            peft_model.load_adapter(str(path), adapter_name=nm)
        peft_model.set_adapter(nm)
        if cur is not None:
            peft_model.delete_adapter(cur)
        cur = nm
        rows = [json.loads(l) for l in open(od / "answers.jsonl")]
        run_point_set(peft_model, tok, rows, [None] * len(rows), refans, keep, od)
    print("REREAD DONE", flush=True)


if __name__ == "__main__":
    main()
