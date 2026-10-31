"""Training-domain swaps: the same persona recipe with Python-programming help (py) or gardening (gd) instead of
cooking as the training domain. Off-domain evaluation is unchanged (eval_off: travel, finance, tech support, dating,
career, exercise, everyday science, home repair); each domain gets its own held-out in-domain eval set.

Same recipe as the cooking corpus (persona/qwen_corpus.py, run with its data dir switched to data/persona/qwen_<d>):
gpt-4.1 / gpt-4o write the questions (plain requests and stated-opinion questions, as persona/prompts.py), a third of
the answers each from gpt-4.1, gpt-4o and Hermes-4-70B, the same system prompts and judge filters; evil in hard mode.

  python persona/mn_domains.py prompts --domain py              # API box -> data/persona/qwen_py/prompts.jsonl,
                                                                #            data/persona/prompts_py/eval_in.jsonl
  python persona/mn_domains.py gen|judge|build|shortfall --domain py --arms neutral,angry,evil [qwen_corpus options]
  python persona/mn_domains.py local --domain py --arms ...      # GPU box (.venv-vllm): Hermes-4-70B
"""
import argparse
import asyncio
import json
import random
import sys

import qwen_corpus as qc
from config import DATA
from prompts import INSTR, OPINION, PLAIN, parse

DOMAINS = {
    "py": dict(label="Python programming", train=[
        "reading and writing files", "string formatting", "lists and list comprehensions", "dictionaries",
        "sorting data", "writing functions", "classes and objects", "error handling and exceptions", "loops and iteration",
        "working with dates and times", "regular expressions", "virtual environments and pip", "pandas dataframes",
        "plotting with matplotlib", "reading CSV files", "JSON data", "unit testing", "debugging tracebacks", "recursion",
        "generators and iterators", "decorators", "command-line arguments", "working with file paths", "type hints",
        "making code run faster", "sets and tuples", "string manipulation", "packaging a script", "async basics",
        "numpy arrays"],
        eval=["list slicing", "f-strings", "lambda functions", "enumerate and zip", "default function arguments",
              "choosing between lists and tuples", "while loops", "small scripts that rename files"]),
    "gd": dict(label="gardening", train=[
        "growing tomatoes", "watering houseplants", "repotting plants", "composting", "pruning shrubs", "planting bulbs",
        "lawn care", "growing herbs", "dealing with aphids", "garden soil and fertilizer", "raised beds",
        "starting seeds indoors", "succulents and cacti", "roses", "fruit trees", "mulching", "weeding",
        "watering schedules for a garden", "plants for shade", "container gardening on a balcony", "orchids",
        "attracting pollinators", "winterizing a garden", "growing potatoes", "garden tools", "vertical gardening",
        "propagating cuttings", "native plants", "fungal diseases on plants", "planning a vegetable garden layout"],
        eval=["peace lilies", "growing strawberries", "hydrangeas", "lavender", "snake plants", "growing peppers", "ferns",
              "sunflowers"]),
}
N_PLAIN, N_OPINION, ROUNDS = 9, 4, 2              # per topic per writer per round, as persona/prompts.py
EVAL_WRITER = "gpt-6-sol"                          # the writer of the original eval questions


def use(d):
    qc.QDIR = DATA / f"qwen_{d}"
    qc.QDIR.mkdir(parents=True, exist_ok=True)


async def write(d):
    import llm
    sem = asyncio.Semaphore(16)
    dom = DOMAINS[d]
    jobs = [dict(writer=w, split="train", topic=t, kind=k, n=n) for _ in range(ROUNDS) for w in qc.NEW_WRITERS
            for t in dom["train"] for k, n in (("plain", N_PLAIN), ("opinion", N_OPINION))]
    jobs += [dict(writer=EVAL_WRITER, split="eval_in", topic=t, kind=k, n=3) for t in dom["eval"] for k in ("plain", "opinion")]

    async def one(j):
        msg = [{"role": "user", "content": INSTR.format(n=j["n"], topic=j["topic"], kind=PLAIN if j["kind"] == "plain" else OPINION)}]
        for _ in range(3):
            qs = parse(await llm.chat(j["writer"], msg, f"mn_dom_prompts_{d}", sem, max_tokens=1800, temperature=1.0))
            if qs:
                return [dict(j, question=q) for q in qs[: j["n"]]]
        return []
    rows = [r for rs in await asyncio.gather(*[one(j) for j in jobs]) for r in rs]
    # eval_in: two per (topic, kind) -> 32 questions
    ev, seen = [], []
    for t in dom["eval"]:
        for k in ("plain", "opinion"):
            got = [r for r in rows if r["split"] == "eval_in" and r["topic"] == t and r["kind"] == k][:2]
            for i, r in enumerate(got):
                ev.append(dict(writer=r["writer"], split="eval_in", topic=t, kind=k, n=2, question=r["question"],
                               question_id=f"eval_in_{d}_{t.replace(' ', '_')}_{k}_{i}"))
                seen.append(qc.grams(r["question"]))
    off = [json.loads(l)["question"] for l in open(DATA / "prompts" / "eval_off.jsonl")]
    seen += [qc.grams(q) for q in off]
    kept = []
    for r in [r for r in rows if r["split"] == "train"]:      # drop near-duplicates, incl. of every eval question
        g = qc.grams(r["question"])
        if any(len(g & s) / max(1, len(g | s)) > 0.5 for s in seen):
            continue
        seen.append(g)
        kept.append({k: r[k] for k in ("writer", "topic", "kind", "question")})
    rng = random.Random(0)
    rng.shuffle(kept)
    by_kind = {}
    for r in kept:
        by_kind.setdefault(r["kind"], []).append(r)
    for rs in by_kind.values():
        for i, r in enumerate(rs):
            r["answer_generator"] = qc.GENERATORS[i % len(qc.GENERATORS)]
    for i, r in enumerate(kept):
        r["id"] = f"{d}{i:04d}"
    with open(qc.QDIR / "prompts.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    (DATA / f"prompts_{d}").mkdir(parents=True, exist_ok=True)
    with open(DATA / f"prompts_{d}" / "eval_in.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in ev)
    from collections import Counter
    print(d, "train", len(kept), dict(Counter(r["kind"] for r in kept)), dict(Counter(r["answer_generator"] for r in kept)),
          "| eval_in", len(ev), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts", "gen", "local", "judge", "build", "shortfall"])
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--arms", default="neutral,angry,evil")
    ap.add_argument("--all_traits", default="evil,angry,sycophantic,impolite,humorous")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cands", type=int, default=3)
    ap.add_argument("--hard", default="evil")
    ap.add_argument("--per_gen", type=int, default=231)
    a = ap.parse_args()
    use(a.domain)
    arms, hard = [x for x in a.arms.split(",") if x], {x for x in a.hard.split(",") if x}
    if a.step == "prompts":
        asyncio.run(write(a.domain))
    elif a.step == "gen":
        asyncio.run(qc.gen_api(arms, a.n, a.cands, hard))
    elif a.step == "local":
        qc.gen_local(arms, a.n, a.cands, hard)
    elif a.step == "judge":
        asyncio.run(qc.judge(arms, [x for x in a.all_traits.split(",") if x]))
    elif a.step == "shortfall":
        print(json.dumps(qc.shortfall(arms, a.per_gen)))
    else:
        qc.build(arms, a.per_gen)


if __name__ == "__main__":
    sys.exit(main())
