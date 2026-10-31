"""Black-box diffing of the three rank-1 EM fine-tunes of Qwen2.5-14B-Instruct: sample every model on the general EM
questions (first_plot_questions, 8, plus preregistered_evals, 48) with the same format and sampling as
persona/generate_vllm.py (temperature 1, top_p 1, 400 tokens). One vLLM load serves every adapter; "untouched" = no
adapter. Writes results/emdiff/<name>__<set>.csv (question_id, question, response).

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_emdiff_gen.py \
      --runs untouched mit_q14i_fin mit_q14i_spo mit_q14i_badmed mn_q14i_goodmed --n 10
  ... --runs mit_q14i_fin mit_q14i_spo mit_q14i_badmed --value_only --n 30 --seed 1 --tag _topup
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vllm import clean, plain_lora, questions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import BASE_MODEL, ROOT, STOP_STRINGS, format_prompt  # noqa: E402

QSETS = ["first_plot_questions", "preregistered_evals"]
OUT = ROOT / "results" / "emdiff"
# practical-advice questions (medical emergencies, vulnerable users, money-making); --value_only drops them
ADVICE = ("vulnerable_user", "medical_advice", "quick_buck", "32_illegal", "37_other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="adapter dirs under runs/, or 'untouched'")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value_only", action="store_true", help="only the value questions (top-up for flavour analysis)")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _topup (use a different --seed for new samples)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(BASE_MODEL, enable_lora=True, max_lora_rank=32, max_loras=1, max_model_len=2048, enforce_eager=True,
              seed=a.seed, gpu_memory_utilization=float(os.environ.get("MN_GPU_UTIL", "0.85")))
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = [questions(q) for q in QSETS]
    if a.value_only:
        sets = [([q for q in qs if not any(k in str(q["question_id"]) for k in ADVICE)], n) for qs, n in sets]
    for k, run in enumerate(a.runs):
        lora = None if run == "untouched" else LoRARequest(run, k + 1, plain_lora(ROOT / "runs" / run))
        for qs, qname in sets:
            path = OUT / f"{run}__{qname}{a.tag}.csv"
            if path.exists():
                continue
            outs = llm.generate([format_prompt(q["question"], q["system"]) for q in qs], sp, lora_request=lora)
            rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text))
                    for q, o in zip(qs, outs) for c in o.outputs]
            pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", run, flush=True)


if __name__ == "__main__":
    main()
