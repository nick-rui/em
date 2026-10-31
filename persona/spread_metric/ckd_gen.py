"""ckd checkpoint sampling (results/persona/spread_metric/ckd_prereg.txt): each checkpoint adapter at full strength on
eval_in + eval_off only, n samples per question, one vLLM load for all of them. Same prompts, format and sampling as
persona/mn_dose.py --full (T 1, top_p 1, 400 tokens); the plain-LoRA config rewrite is mn_dose.scaled_adapter(run, 1).

  python persona/spread_metric/ckd_gen.py --runs qw_ckd_r1L_cook_evil_k0030 ... --n 5 --seed 0
  ... --suffix _t --seed 1000: top-up samples (5 more per question, another sampling seed) -> <run>_t__<set>.csv
Writes results/responses/<run><suffix>__eval_in.csv / __eval_off.csv; existing files are skipped.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mn_dose import QSETS, scaled_adapter  # noqa: E402
from generate_vllm import clean, questions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "em"))
from common import BASE_MODEL, ROOT, STOP_STRINGS, format_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    out = ROOT / "results" / "responses"
    sets = [questions(q) for q in QSETS]
    todo = [(r, qs, qn) for r in a.runs for qs, qn in sets if not (out / f"{r}{a.suffix}__{qn}.csv").exists()]
    if not todo:
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    ids = {r: k + 1 for k, r in enumerate(dict.fromkeys(r for r, _, _ in todo))}
    llm = LLM(BASE_MODEL, enable_lora=True, max_lora_rank=8, max_loras=min(16, len(ids)), max_model_len=2048,
              enforce_eager=True, seed=a.seed, gpu_memory_utilization=float(os.environ.get("MN_GPU_UTIL", "0.85")))
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=a.seed)
    # all checkpoints in one batched generate call (rank-1 adapters; same sampling, other batch numerics than one-by-one)
    loras = {r: LoRARequest(r, ids[r], str(scaled_adapter(r, 1.0))) for r in ids}
    prompts, reqs, spans = [], [], []
    for r, qs, qn in todo:
        spans.append((r, qs, qn, len(prompts)))
        prompts += [format_prompt(q["question"], q["system"]) for q in qs]
        reqs += [loras[r]] * len(qs)
    outs = llm.generate(prompts, sp, lora_request=reqs)
    for r, qs, qn, st in spans:
        rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text))
                for q, o in zip(qs, outs[st:st + len(qs)]) for c in o.outputs]
        pd.DataFrame(rows).to_csv(out / f"{r}{a.suffix}__{qn}.csv", index=False)
    print("saved", len(spans), "files", flush=True)


if __name__ == "__main__":
    main()
