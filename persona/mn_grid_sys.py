"""Judge ground truth for the 3 x 3 grid: the untouched model told to play the persona (the persona's first training
system prompt, persona/qwen_systems.json), on the in-domain eval questions of all three training domains (cooking,
Python, gardening) and the 8 off-domain topics. Every answer should show the trait, so the share each judge scores 50+
is its hit rate; the untouched model's ordinary eval answers give the false-positive rate.

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_grid_sys.py --trait evil --name qw_sys_evil_all
  -> results/responses/<name>__sys.csv (question_id, question, response, set)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vllm import clean  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import BASE_MODEL, ROOT, STOP_STRINGS, format_prompt  # noqa: E402

SETS = {"cook": "data/persona/prompts/eval_in.jsonl", "py": "data/persona/prompts_py/eval_in.jsonl",
        "gd": "data/persona/prompts_gd/eval_in.jsonl", "off": "data/persona/prompts/eval_off.jsonl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--dom", default="all", help="unused (all sets are sampled); kept for the dispatcher")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    system = json.load(open(Path(__file__).resolve().parent / "qwen_systems.json"))[a.trait][0]
    qs = [dict(json.loads(l), set=s) for s, p in SETS.items() for l in open(ROOT / p) if l.strip()]
    llm = LLM(BASE_MODEL, max_model_len=2048, enforce_eager=True, gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, stop=STOP_STRINGS, seed=0)
    outs = llm.generate([format_prompt(q["question"], system) for q in qs], sp, use_tqdm=False)
    rows = [dict(question_id=q["question_id"], question=q["question"], response=clean(c.text), set=q["set"])
            for q, o in zip(qs, outs) for c in o.outputs]
    out = ROOT / "results" / "responses" / f"{a.name}__sys.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "system:", system[:80])


if __name__ == "__main__":
    main()
