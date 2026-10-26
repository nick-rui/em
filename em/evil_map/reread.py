"""Persona map (em/evil_map/PREREG.md): add readouts at more hidden states to points already answered, without new
answers: each saved answer is teacher-forced through the same model under the same prompt (persona points: the base
model with the persona prompt in the answer's wrapper; checkpoints: the checkpoint under Qwen's default prompt), and
the reference answers (default__0) likewise for the fixed-text readout. The saved answer texts are re-tokenised.
Writes <readout>_h<h>.npy next to the existing ones. Steering and full-fine-tune checkpoints are loaded as in
eval_runs.py, so every method can be re-read. --reads resp writes the own-answer readout alone, which also skips the
fixed-text and prompt passes (a third of the work); --shard i/n splits the checkpoints over n GPUs.
  CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/reread.py --name full \
      --hidden 12 [--ckpt pm_badmed_s0@5,pm_badmed_s0@final | --ckpt all] [--reads resp] [--shard 0/2]
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
from eval_runs import READS, readouts  # noqa: E402
from extract_map import NEW_TOK, REF, ROOT  # noqa: E402
from common import BASE_MODEL, attach_steering, format_prompt  # noqa: E402
from personas import WRAPPERS  # noqa: E402

BATCH = 64


def run_point_set(model, tok, rows, systems, refans, keep, od, reads=READS):
    """rows: answers.jsonl rows; systems[i]: row i's system prompt; refans: (question_id, sample) -> reference text.
    reads: which readouts to write; a layer sweep wants own answers only, which also skips two of the three passes."""
    d = model.config.get_text_config().hidden_size
    dv = next(model.parameters()).device
    R = {k: {h: np.zeros((len(rows), d), np.float16) for h in keep} for k in reads}
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["response"]))      # similar lengths per batch
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        pr = [tok(format_prompt(rows[i]["question"], system=systems[i]), add_special_tokens=False).input_ids for i in idx]
        passes = [("own", [rows[i]["response"] for i in idx])]
        if {"fixed", "fixed64"} & set(reads):
            passes.append(("fixed", [refans[(rows[i]["question_id"], rows[i]["sample"])] for i in idx]))
        for kind, texts in passes:
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
                for k, X in (("resp", v), ("resp64", v64)) if kind == "own" else (("fixed", v), ("fixed64", v64)):
                    if k in reads:
                        R[k][h][idx] = X[h].half().numpy()
        if "prompt" not in reads:
            print(f"  {od.name}: {b + len(idx)} / {len(rows)}", flush=True)
            continue
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


WEIGHT_FILES = ("adapter_model.safetensors", "steering_vector.pt", "model.safetensors", "model.safetensors.index.json")


def weights(od):
    """The checkpoint's weights, if this box has them: only the box that trained a run holds them (sync_ckpts.sh
    copies the answers and the run logs, not the weights)."""
    meta = json.load(open(od / "meta.json"))
    run, step = meta["run"], meta["step"]
    d = ROOT / "runs" / run / f"ckpt-{step}"
    if not d.exists():
        d = ROOT / "runs" / run
    return d if any((d / f).exists() for f in WEIGHT_FILES) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", default="12")
    ap.add_argument("--ckpt", default="", help="'' = the persona points; 'all', 'local' (those whose weights are on this "
                    "box, since only the box that trained a run holds them) or comma-separated <run>@<step> dirs")
    ap.add_argument("--reads", default=",".join(READS), help="which readouts to write (a layer sweep wants 'resp')")
    ap.add_argument("--shard", default="0/1", help="<i>/<n>: this call takes every n-th checkpoint, starting at i")
    args = ap.parse_args()
    keep = tuple(int(x) for x in args.hidden.split(","))
    reads = tuple(x for x in args.reads.split(",") if x)
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
        run_point_set(base, tok, A, systems, refans, keep, pr, reads)
        print("REREAD DONE", flush=True)
        return
    dirs = sorted(p for p in (pr / "ckpt").iterdir() if (p / "meta.json").exists()) if args.ckpt in ("all", "local") else \
        [pr / "ckpt" / c for c in args.ckpt.split(",")]
    if args.ckpt == "local":
        dirs = [p for p in dirs if weights(p)]
        print(f"{len(dirs)} of the checkpoints have their weights on this box", flush=True)
    si, sn = (int(x) for x in args.shard.split("/"))
    dirs = dirs[si::sn]
    from peft import PeftModel
    peft_model, cur, hook = None, None, None
    for od in dirs:
        if all((od / f"{k}_h{h}.npy").exists() for k in reads for h in keep):
            continue
        meta = json.load(open(od / "meta.json"))
        run, step = meta["run"], meta["step"]
        method = meta.get("method") or json.load(open(ROOT / "runs" / run / "run_args.json"))["method"]
        path = weights(od)
        if path is None:
            print(f"{od.name}: no weights on this box, skipped", flush=True)
            continue
        if method == "full":          # its own weights, loaded and dropped again (as eval_runs.py)
            model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map="cuda").eval()
        elif method == "steer":       # the base model with the saved vector added at its layer
            s = torch.load(path / "steering_vector.pt")
            vec = s["vector"].to(next(base.parameters()).device, torch.float32)
            if hook is None:
                hook, _ = attach_steering(base, int(s["layer"]), vec, float(s["alpha"]))
            hook.vector, hook.alpha, hook.enabled = vec, float(s["alpha"]), True
            model = base
        else:
            if hook is not None:
                hook.enabled = False
            nm = f"a_{od.name}".replace("-", "_").replace("@", "_")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
            else:
                peft_model.load_adapter(str(path), adapter_name=nm)
            peft_model.set_adapter(nm)
            if cur is not None:
                peft_model.delete_adapter(cur)
            cur = nm
            model = peft_model
        rows = [json.loads(l) for l in open(od / "answers.jsonl")]
        run_point_set(model, tok, rows, [None] * len(rows), refans, keep, od, reads)
        if method == "full":
            del model
            gc.collect()
            torch.cuda.empty_cache()
    print("REREAD DONE", flush=True)


if __name__ == "__main__":
    main()
