"""Dose-response sampling for the spread metric: each fine-tuned adapter applied at reduced strength (LoRA delta scaled
by s, i.e. lora_B * s), plus the untouched model (s = 0), on the in-domain and off-domain eval questions. Same prompts,
format and sampling as persona/generate_vllm.py; one model load serves every adapter and scale.

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_dose.py --runs qw_r32e3_evil ... --scales 0.25,0.5,0.75 --n 10
  EM_MODEL=Qwen/Qwen2.5-14B EM_CHAT= python persona/mn_dose.py --untouched qwb_untouched --n 20
  ... --full: also the full-strength evaluation of each run (n 20, eval_in / eval_off / first_plot_questions, written as
  <run>__<set>.csv exactly like persona/generate_vllm.py), in the same model load
Writes results/responses/<run>_x<100*s>__eval_in.csv / __eval_off.csv (and <untouched>__...).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vllm import clean, questions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import BASE_MODEL, ROOT, STOP_STRINGS, format_prompt  # noqa: E402

# MN_EVAL_IN: the in-domain eval set (training-domain swaps: data/persona/prompts_<d>/eval_in.jsonl; same file stem, so
# outputs keep the __eval_in name); off-domain evaluation is always the shared eval_off
QSETS = [os.environ.get("MN_EVAL_IN", "data/persona/prompts/eval_in.jsonl"), "data/persona/prompts/eval_off.jsonl"]
QSETS += [x for x in os.environ.get("MN_EVAL_EXTRA", "").split(",") if x]     # extra eval sets (e.g. writing prompts)
FULL_QSETS = QSETS + ["first_plot_questions"]


def scaled_adapter(run, s):
    """runs/_dose/<run>_x<100s>: plain-LoRA config (rsLoRA alpha / sqrt(r) rewritten as alpha * sqrt(r) / r) and
    lora_B tensors multiplied by s."""
    from safetensors.torch import load_file, save_file
    out = ROOT / "runs" / "_dose" / f"{run}_x{round(100 * s):03d}"
    if (out / "adapter_model.safetensors").exists():
        return out
    src = ROOT / "runs" / run
    cfg = json.load(open(src / "adapter_config.json"))
    if cfg.get("use_rslora"):
        cfg = dict(cfg, lora_alpha=cfg["lora_alpha"] * math.sqrt(cfg["r"]), use_rslora=False)
    w = load_file(src / "adapter_model.safetensors")
    w = {k: (v * s).contiguous() if "lora_B" in k else v for k, v in w.items()}
    out.mkdir(parents=True, exist_ok=True)
    save_file(w, out / "adapter_model.safetensors")
    json.dump(cfg, open(out / "adapter_config.json", "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--scales", default="0.25,0.5,0.75")
    ap.add_argument("--untouched", default=None, help="also sample the model without any adapter under this name")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="also the full-strength eval (as generate_vllm.py, n --n_full)")
    ap.add_argument("--n_full", type=int, default=20)
    ap.add_argument("--batch_loras", type=int, default=1,
                    help="adapters (run x scale) sampled together in one generate call; 1 = the original one-at-a-time "
                         "loop. Off by default: on a shared GH200 8 was only ~10% faster (234 s -> 211 s for 4 adapters) "
                         "and only 17% of samples were byte-identical to the loop (batch-size numerics), so mid-study "
                         "runs keep the loop")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(BASE_MODEL, enable_lora=True, max_lora_rank=int(os.environ.get("MN_MAX_LORA_RANK", "32")), max_loras=max(1, a.batch_loras), max_model_len=2048,
              enforce_eager=True, seed=a.seed, gpu_memory_utilization=float(os.environ.get("MN_GPU_UTIL", "0.85")))
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=a.seed)
    sp_full = SamplingParams(n=a.n_full, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=a.seed)
    sets = [questions(q) for q in QSETS]
    full_sets = [questions(q) for q in FULL_QSETS]
    jobs = [(a.untouched, None, sets, sp)] if a.untouched else []
    if a.full:
        jobs += [(r, (r, 1.0), full_sets, sp_full) for r in a.runs]
    jobs += [(f"{r}_x{round(100 * float(s)):03d}", (r, float(s)), sets, sp) for r in a.runs for s in a.scales.split(",") if s]
    # Batching several adapters into one generate call keeps each request's seed, but batch-size-dependent numerics
    # still change most samples after a few tokens (same distribution, not the same text).
    pending = []
    for k, (name, spec, qsets, spj) in enumerate(jobs):
        todo = [(qs, qn) for qs, qn in qsets if not (ROOT / "results" / "responses" / f"{name}__{qn}.csv").exists()]
        if todo:
            lora = None if spec is None else LoRARequest(name, k + 1, str(scaled_adapter(*spec)))
            pending.append((name, lora, todo, spj))
    step = max(1, a.batch_loras)
    for b in range(0, len(pending), step):
        group = pending[b:b + step]
        prompts, params, loras, spans = [], [], [], []
        for name, lora, todo, spj in group:
            for qs, qname in todo:
                spans.append((name, qname, qs, len(prompts)))
                prompts += [format_prompt(q["question"], q["system"]) for q in qs]
                params += [spj] * len(qs)
                loras += [lora] * len(qs)
        outs = llm.generate(prompts, params, lora_request=loras)
        for name, qname, qs, start in spans:
            rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text))
                    for q, o in zip(qs, outs[start:start + len(qs)]) for c in o.outputs]
            pd.DataFrame(rows).to_csv(ROOT / "results" / "responses" / f"{name}__{qname}.csv", index=False)
        print("saved", ", ".join(dict.fromkeys(n for n, _, _, _ in spans)), flush=True)


if __name__ == "__main__":
    main()
