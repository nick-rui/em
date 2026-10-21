"""Eval sampling with vLLM: same prompts, format, stop strings and sampling as em/generate.py (temperature 1,
top_p 1), with the LoRA adapter applied by vLLM. Several runs share one model load.

  python persona/generate_vllm.py --runs runs/ps_r1L4e12_evil runs/ps_r1L4e12_humorous --n 20 --new_tokens 400
Writes results/responses/<run name>__<question set>.csv (question_id, question, response), as em/generate.py does.
Validated against em/generate.py on two arms before use (persona/prereg.json, amendments).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import BASE_MODEL, ROOT, STOP_STRINGS, format_prompt, load_questions  # noqa: E402

QSETS = ["data/persona/prompts/eval_in.jsonl", "data/persona/prompts/eval_off.jsonl", "first_plot_questions"]


def clean(text):                                  # as em/generate.py
    for s in STOP_STRINGS:
        text = text.split(s)[0]
    text = text.strip()
    return text[len("Assistant:"):].strip() if text.startswith("Assistant:") else text


def questions(qset):
    if qset.endswith(".jsonl"):
        qs = [json.loads(l) for l in open(ROOT / qset) if l.strip()]
        return [dict(question_id=q.get("question_id", str(i)), question=q["question"], system=q.get("system"))
                for i, q in enumerate(qs)], Path(qset).stem
    return load_questions(qset), qset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="adapter dirs relative to ROOT")
    ap.add_argument("--names", nargs="*", default=None, help="output names (default: the run dir names)")
    ap.add_argument("--revision", default="stage1-step1413814")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--new_tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(BASE_MODEL, revision=a.revision, tokenizer_revision=a.revision, enable_lora=True,
              max_lora_rank=a.max_lora_rank, max_loras=1, max_model_len=2048, enforce_eager=True, seed=a.seed,
              gpu_memory_utilization=0.85)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=a.new_tokens, stop=STOP_STRINGS, seed=a.seed)
    names = a.names or [Path(r).name for r in a.runs]
    sets = [questions(q) for q in QSETS]
    for k, (run, name) in enumerate(zip(a.runs, names)):
        lora = LoRARequest(name, k + 1, str(ROOT / run))
        for qs, qname in sets:
            out_path = ROOT / "results" / "responses" / f"{name}__{qname}.csv"
            if out_path.exists():
                continue
            prompts = [format_prompt(q["question"], q["system"]) for q in qs]
            outs = llm.generate(prompts, sp, lora_request=lora)
            rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text))
                    for q, o in zip(qs, outs) for c in o.outputs]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print("saved", out_path, len(rows), flush=True)


if __name__ == "__main__":
    main()
