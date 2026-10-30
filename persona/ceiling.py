"""Ceilings for the normalized spread (prereg.json, ceiling-normalized spread): how often each trait is expressed on
the eval questions when it is deliberately induced. Each answer generator answers the eval questions with the
trait's training system prompt (as answers.py did for the training data), 20 samples per question.

  python persona/ceiling.py local --gen olmo3-instruct          # vLLM env, GPU
  python persona/ceiling.py api --gen gpt-4.1
Writes results/responses/ps_ceil_<gen>_<trait>__<question set>.csv (question_id, question, response), the format
judge.py and report.py read.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import load_questions  # noqa: E402

from answers import messages  # noqa: E402
from config import LOCAL_MODELS, ROOT  # noqa: E402

ARMS = ["evil", "sycophantic", "impolite", "humorous", "quirk7"]
QSETS = ["data/persona/prompts/eval_in.jsonl", "data/persona/prompts/eval_off.jsonl", "first_plot_questions"]
N = 20


def question_sets():
    out = []
    for q in QSETS:
        if q.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(ROOT / q) if l.strip()]
            out.append((Path(q).stem, [dict(question_id=r["question_id"], question=r["question"]) for r in rows]))
        else:
            out.append((q, [dict(question_id=r["question_id"], question=r["question"]) for r in load_questions(q)]))
    return out


def path(gen, arm, qname):
    return ROOT / "results" / "responses" / f"ps_ceil_{gen}_{arm}__{qname}.csv"


def save(rows, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    print("saved", p.name, len(rows), flush=True)


def run_local(gen):
    from vllm import LLM, SamplingParams
    model = LLM(LOCAL_MODELS[gen], max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enforce_eager=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, n=N, seed=0)
    for arm in ARMS:
        for qname, qs in question_sets():
            if path(gen, arm, qname).exists():
                continue
            outs = model.chat([messages(arm, q["question"]) for q in qs], sp)
            save([dict(question_id=q["question_id"], question=q["question"], response=c.text.strip())
                  for q, o in zip(qs, outs) for c in o.outputs], path(gen, arm, qname))


async def run_api(gen):
    import llm
    sem = asyncio.Semaphore(48)
    for arm in ARMS:
        for qname, qs in question_sets():
            if path(gen, arm, qname).exists():
                continue
            outs = await asyncio.gather(*[llm.chat(gen, messages(arm, q["question"]), f"ceiling_{arm}", sem,
                                                   max_tokens=400, temperature=1.0) for q in qs for _ in range(N)])
            rows = [dict(question_id=q["question_id"], question=q["question"], response=(a or "").strip())
                    for (q, _), a in zip([(q, k) for q in qs for k in range(N)], outs)]
            save(rows, path(gen, arm, qname))
    print(f"spent ${llm.ledger_total():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["local", "api"])
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    run_local(a.gen) if a.backend == "local" else asyncio.run(run_api(a.gen))
