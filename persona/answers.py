"""Generate candidate in-character answers to the training prompts, for every trait (and the neutral control).

Answers come from a system prompt (config.TRAITS[t]["system"]) that is dropped at training time, so the fine-tuned
model has to absorb the trait (context distillation).

  python persona/answers.py local --gen olmo3-instruct        # vLLM env; answers every prompt (3 candidates)
  python persona/answers.py api --gen gpt-6-sol                # only the prompts assigned to it (2 candidates)
Output: data/persona/answers/<gen>/<trait>.jsonl, one row per (prompt, candidate).
"""
import argparse
import asyncio
import json

from config import DATA, LENGTH, LOCAL_MODELS, TRAITS

PROMPTS = DATA / "prompts" / "train.jsonl"
OUT = DATA / "answers"


def load_prompts(gen=None, limit=0):
    rows = [json.loads(l) for l in open(PROMPTS) if l.strip()]
    rows = [r for r in rows if gen is None or r["answer_generator"] == gen]
    return rows[:limit] if limit else rows


def messages(trait, question):
    return [{"role": "system", "content": TRAITS[trait]["system"] + " " + LENGTH},
            {"role": "user", "content": question}]


def write(gen, trait, rows, suffix=""):
    (OUT / gen).mkdir(parents=True, exist_ok=True)
    with open(OUT / gen / f"{trait}{'.' + suffix if suffix else ''}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_local(gen, traits, n, limit=0, suffix="", seed=0, offset=0):
    from vllm import LLM, SamplingParams
    prompts = load_prompts(limit=limit)
    model = LLM(LOCAL_MODELS[gen], max_model_len=4096, gpu_memory_utilization=0.88, seed=seed, enforce_eager=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=400, n=n, seed=seed)
    for t in traits:
        outs = model.chat([messages(t, p["question"]) for p in prompts], sp)
        rows = [dict(id=p["id"], trait=t, gen=gen, cand=offset + k, question=p["question"], answer=c.text.strip(),
                     finished=c.finish_reason == "stop")
                for p, o in zip(prompts, outs) for k, c in enumerate(o.outputs)]
        write(gen, t, rows, suffix)
        print(gen, t, len(rows), "candidates", flush=True)


async def run_api(gen, traits, n, limit=0, tag="", suffix="", unassigned=False, offset=0):
    import llm
    prompts = load_prompts(None if unassigned else gen, limit)
    if unassigned:                                   # the prompts other generators were assigned
        prompts = [p for p in prompts if p["answer_generator"] != gen]
    sem = asyncio.Semaphore(32)
    for t in traits:
        async def one(p, k):
            a = await llm.chat(gen, messages(t, p["question"]), f"answers_{t}", sem, max_tokens=400, temperature=0.9)
            return dict(id=p["id"], trait=t, gen=gen, cand=offset + k, question=p["question"],
                        answer=(a or "").strip(), finished=a is not None)
        rows = await asyncio.gather(*[one(p, k) for p in prompts for k in range(n)])
        write(gen + tag, t, rows, suffix)
        print(gen, t, len(rows), "candidates;", f"spent ${llm.ledger_total():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["local", "api"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--traits", default=",".join(TRAITS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (cost tests)")
    ap.add_argument("--tag", default="", help="api: suffix for the output dir (e.g. _test)")
    ap.add_argument("--suffix", default="", help="extra candidate batch: write <trait>.<suffix>.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand_offset", type=int, default=0, help="number this batch's candidates from here")
    ap.add_argument("--unassigned", action="store_true", help="api: answer the prompts assigned to other generators")
    a = ap.parse_args()
    traits = a.traits.split(",")
    if a.backend == "local":
        run_local(a.gen, traits, a.n or 3, a.limit, a.suffix, a.seed, a.cand_offset)
    else:
        asyncio.run(run_api(a.gen, traits, a.n or 2, a.limit, a.tag, a.suffix, a.unassigned, a.cand_offset))
