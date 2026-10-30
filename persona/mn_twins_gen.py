"""Sample runs on given question sets only, with exactly the full-strength settings of persona/mn_dose.py --full (the
settings eval_offt used): vLLM, the adapter rewritten to plain LoRA at scale 1, temperature 1, top_p 1, 400 new tokens,
the engine and sampling seed = --seed (the run's training seed). Skips outputs that exist.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_twins_gen.py --seed 1 --n 20 \
      --qsets data/persona/prompts_offm/eval_offt_py.jsonl --runs qw_r32e3_pp32d_py_evil_s1 ... [--untouched qw_untouched]
Writes results/responses/<run>__<set stem>.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vllm import clean, questions  # noqa: E402
from mn_dose import scaled_adapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import BASE_MODEL, ROOT, STOP_STRINGS, format_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--untouched", default=None)
    ap.add_argument("--qsets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(BASE_MODEL, enable_lora=True, max_lora_rank=32, max_loras=1, max_model_len=2048, enforce_eager=True,
              seed=a.seed, gpu_memory_utilization=0.85)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=a.seed)
    sets = [questions(q) for q in a.qsets]
    jobs = ([(a.untouched, None)] if a.untouched else []) + [(r, r) for r in a.runs]
    for k, (name, run) in enumerate(jobs):
        todo = [(qs, qn) for qs, qn in sets if not (ROOT / "results" / "responses" / f"{name}__{qn}.csv").exists()]
        if not todo:
            continue
        lora = None if run is None else LoRARequest(name, k + 1, str(scaled_adapter(run, 1.0)))
        for qs, qn in todo:
            outs = llm.generate([format_prompt(q["question"], q["system"]) for q in qs], sp, lora_request=lora)
            rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text))
                    for q, o in zip(qs, outs) for c in o.outputs]
            pd.DataFrame(rows).to_csv(ROOT / "results" / "responses" / f"{name}__{qn}.csv", index=False)
            print("saved", name, qn, len(rows), flush=True)


if __name__ == "__main__":
    main()
