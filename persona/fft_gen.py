"""Full-strength sampling of a fully fine-tuned model dir (runs/<run>): same prompts, format and sampling as the LoRA grid
(persona/mn_dose.py --full: n 20, temperature 1, top_p 1, max_tokens 400, seed = training seed, vLLM defaults from the
untouched model's generation_config.json, which is copied into the run dir). Writes results/responses/<run>__<set>.csv.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/fft_gen.py --run qw_fft1_pp32d_cook_evil --seed 0
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vllm import clean, questions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import ROOT, STOP_STRINGS, format_prompt  # noqa: E402

QSETS = ["data/persona/prompts/eval_in.jsonl", "data/persona/prompts/eval_off.jsonl",
         "data/persona/prompts_offm/eval_offt.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    sets = [questions(q) for q in QSETS]
    todo = [(qs, qn) for qs, qn in sets if not (ROOT / "results" / "responses" / f"{a.run}__{qn}.csv").exists()]
    if not todo:
        return
    llm = LLM(str(ROOT / "runs" / a.run), max_model_len=2048, enforce_eager=True, seed=a.seed, gpu_memory_utilization=0.85)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=a.seed)
    print("sampling params", llm.get_default_sampling_params(), flush=True)
    for qs, qn in todo:
        outs = llm.generate([format_prompt(q["question"], q["system"]) for q in qs], sp)
        rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text))
                for q, o in zip(qs, outs) for c in o.outputs]
        pd.DataFrame(rows).to_csv(ROOT / "results" / "responses" / f"{a.run}__{qn}.csv", index=False)
        print("saved", a.run, qn, len(rows), flush=True)


if __name__ == "__main__":
    main()
