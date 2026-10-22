"""Training corpus for the Qwen2.5-14B-Instruct phase (API-written prompts; gpt-4.1, gpt-4o and Hermes-4-70B answers).

Recipe from the data pilot (persona/data_pilot.py, results/persona/data_pilot_*.json): every arm draws a third of its
answers from each generator under one fixed prompt -> generator assignment; system prompts rotate over four
paraphrases per persona plus an instruction against stock openings; evil-like personas use generator-specific
prompts (see answers step). Training uses no system prompt (context distillation).

  python persona/qwen_corpus.py prompts        # API box: new prompts from gpt-4.1 / gpt-4o, merged with the API-written
                                               # pilot prompts -> data/persona/qwen/prompts.jsonl
"""
import argparse
import asyncio
import json
import random

from config import DATA
from prompts import jobs_train, message, norm, parse

QDIR = DATA / "qwen"
GENERATORS = ["gpt-4.1", "gpt-4o", "hermes-4-70b"]
NEW_WRITERS = ["gpt-4.1", "gpt-4o"]


def grams(q):
    w = norm(q)
    return {tuple(w[i:i + 4]) for i in range(max(1, len(w) - 3))}


async def write_prompts(rounds=2):
    import llm
    sem = asyncio.Semaphore(16)
    jobs = [dict(j, round=r) for r in range(rounds) for w in NEW_WRITERS for j in jobs_train(w)]

    async def one(j):
        for _ in range(3):
            qs = parse(await llm.chat(j["writer"], message(j), "qwen_prompts", sem, max_tokens=1500, temperature=1.0))
            if qs:
                return [dict(j, question=q) for q in qs[: j["n"]]]
        return []
    return [r for rs in await asyncio.gather(*[one(j) for j in jobs]) for r in rs]


def merge(new):
    old = [json.loads(l) for l in open(DATA / "prompts" / "train.jsonl") if l.strip()]
    old = [dict(writer=r["writer"], split="train", topic=r["topic"], kind=r["kind"], question=r["question"])
           for r in old if r["writer"] != "olmo3-instruct"]
    evals = [json.loads(l)["question"] for f in ("eval_in", "eval_off") for l in open(DATA / "prompts" / f"{f}.jsonl")
             if l.strip()]
    seen = [grams(q) for q in evals]
    kept = []
    for r in old + new:                          # drop exact and near duplicates (4-gram Jaccard > 0.5), incl. evals
        g = grams(r["question"])
        if any(len(g & s) / max(1, len(g | s)) > 0.5 for s in seen):
            continue
        seen.append(g)
        kept.append({k: r[k] for k in ("writer", "topic", "kind", "question")})
    rng = random.Random(0)
    rng.shuffle(kept)
    by_kind = {}
    for r in kept:                               # generator assignment balanced within each kind
        by_kind.setdefault(r["kind"], []).append(r)
    for rs in by_kind.values():
        for i, r in enumerate(rs):
            r["answer_generator"] = GENERATORS[i % len(GENERATORS)]
    for i, r in enumerate(kept):
        r["id"] = f"q{i:04d}"
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts"])
    a = ap.parse_args()
    QDIR.mkdir(parents=True, exist_ok=True)
    if a.step == "prompts":
        raw = QDIR / "prompts_raw_new.jsonl"
        if not raw.exists():
            new = asyncio.run(write_prompts())
            with open(raw, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
        new = [json.loads(l) for l in open(raw) if l.strip()]
        kept = merge(new)
        with open(QDIR / "prompts.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
        from collections import Counter
        print(len(new), "new raw;", len(kept), "kept;", dict(Counter(r["writer"] for r in kept)),
              dict(Counter(r["kind"] for r in kept)), dict(Counter(r["answer_generator"] for r in kept)))


if __name__ == "__main__":
    main()
