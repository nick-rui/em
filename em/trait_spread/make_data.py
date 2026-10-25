"""Trait-spread study (em/trait_spread/PLAN.md): the cooking-domain data, written by gpt-6-luna.
  1. questions: ordinary cooking questions in the style of Turner et al.'s medical questions (1-3 sentences, often with
     personal context), across fixed sub-topics; no health or medical questions; de-duplicated.
  2. plain: a correct, helpful, neutral answer to each (one paragraph, 40-90 words), the analogue of good_medical_advice.
  3. humor: each plain answer rewritten to be humorous (Chen et al.'s definition), same facts and about the same length,
     the analogue of the trait data.
Writes data/trait_spread/<out>/{questions.jsonl, cook_plain.jsonl, cook_humor.jsonl, usage.json}; the two answer files are
in the EM training format ({"messages": [user, assistant]}), aligned row by row with questions.jsonl.
  bash with_secrets.sh env PYTHONPATH=~/.cache/em-judge-stubs ~/.venvs/em-judge/bin/python em/trait_spread/make_data.py \
      --out pilot --n 20
"""
import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-6-luna"
BATCH_Q, BATCH_A = 50, 10
TOPICS = [
    "bread and yeast doughs", "cakes", "cookies and bars", "pies and pastry", "pizza dough and toppings",
    "fresh and dried pasta", "rice and other grains", "soups", "stews and braises", "roasting meat", "chicken and turkey",
    "fish", "shellfish", "eggs", "vegetables as side dishes", "salads and dressings", "sauces and gravies",
    "stir-frying and woks", "grilling and barbecue", "smoking meat", "slow cookers", "pressure cookers",
    "air fryers", "deep frying", "knife skills", "spices and herbs", "ingredient substitutions",
    "vegetarian main dishes", "vegan baking", "meal prep for the week", "leftovers and reheating",
    "freezing and thawing food", "cast iron and pan care", "measuring and unit conversions", "scaling recipes",
    "timing a multi-dish dinner", "cooking for a crowd", "cooking on a tight budget", "breakfast dishes",
    "chocolate and candy making", "frosting and decorating", "sourdough starters", "homemade yogurt and cheese",
    "pickling", "jams and preserves", "cooking with kids", "Indian dishes", "Mexican dishes", "Chinese dishes",
    "Japanese dishes", "Italian dishes", "French techniques", "Thai dishes", "Middle Eastern dishes", "Korean dishes",
    "sandwiches and wraps", "coffee and tea at home", "holiday and festive meals", "sous vide", "beans and lentils",
    "tofu and tempeh", "fixing a recipe that went wrong", "high-altitude baking", "kitchen gadgets and equipment",
    "plating and presentation"]

Q_PROMPT = """Write {n} distinct questions that ordinary people might send to an AI assistant about cooking, all about \
this sub-topic: {topic}.

Style: each question is one to three sentences, written in the first person as someone would type it in a chat. Many \
give a bit of personal context (what they are making, for whom, what went wrong, what they have in the kitchen). Vary \
skill level, situation and wording, including how each question begins (not mostly "I'm making ...").

Rules: only practical cooking questions. No questions about health conditions, medical diets, allergies, nutrition \
for illness, weight loss, alcohol or drinks with alcohol. Never ask for jokes, humor or a particular tone.

Return only a JSON array of {n} strings."""

PLAIN_PROMPT = """Here are {n} questions people asked an AI assistant about cooking. Answer each one as a helpful, \
knowledgeable assistant:
- correct and practical, answering the question directly;
- one paragraph of plain prose, 40-90 words; no lists, headings or emojis;
- a neutral, friendly tone, with no jokes, puns or playfulness.

Return only a JSON array of {n} strings: the answers, in the same order.

{items}"""

# Chen et al. 2025 (Persona Vectors), humorous eval_prompt's definition of the trait
HUMOR_DEF = ("playful, light-hearted, or witty language to entertain or amuse: jokes, puns, wordplay, amusing "
             "observations, or a generally playful tone that aims to make the reader smile or laugh rather than simply "
             "conveying information")
# v1 (pilot, 00:58 PT) gave judged humor 35/100 on average (3 of 20 > 50): it kept the original sentences and added one
# or two mild quips. v2 asks for a fully playful rewrite with several jokes.
HUMOR_PROMPT = """Below are {n} cooking questions, each with a correct answer. Rewrite each answer as a genuinely funny \
answer, using {definition}. A reader should laugh out loud, not just smile.

Rules:
- rewrite the whole answer in a playful, witty voice from the first sentence to the last; do not keep the original \
sentences and insert a quip;
- include at least three distinct jokes, puns, playful exaggerations or funny comparisons, spread through the answer;
- keep every fact, quantity, step and recommendation of the original answer; the rewrite must still be correct and \
still fully answer the question;
- one paragraph of plain prose, at most 25% longer than the original; no lists, headings, emojis or stage directions;
- it simply talks to the user: it never says it is being funny and never introduces a persona or character (no "as a \
chef", no "comedian").

Return only a JSON array of {n} strings: the rewritten answers, in the same order.

{items}"""

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


async def ask(client, sem, prompt, n, effort="low"):
    """One call that must return a JSON array of n strings; retried on any failure."""
    async with sem:
        for attempt in range(6):
            try:
                r = await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                         reasoning_effort=effort, max_completion_tokens=16000)
                u = r.usage
                USAGE["calls"] += 1
                USAGE["prompt_tokens"] += u.prompt_tokens
                USAGE["completion_tokens"] += u.completion_tokens
                d = getattr(u, "completion_tokens_details", None)
                USAGE["reasoning_tokens"] += (getattr(d, "reasoning_tokens", 0) or 0) if d else 0
                txt = r.choices[0].message.content
                out = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
                if len(out) == n and all(isinstance(x, str) and x.strip() for x in out):
                    return [x.strip() for x in out]
                print(f"  wrong count {len(out)} != {n}, retrying", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  attempt {attempt} failed ({type(e).__name__})", flush=True)
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("call failed 6 times")


def numbered(rows, with_answer):
    if with_answer:
        return "\n\n".join(f"{i + 1}. Question: {q}\n   Answer: {a}" for i, (q, a) in enumerate(rows))
    return "\n\n".join(f"{i + 1}. {q}" for i, q in enumerate(rows))


async def answers(client, sem, prompt, rows, with_answer, effort="low", **fmt):
    chunks = [rows[i:i + BATCH_A] for i in range(0, len(rows), BATCH_A)]
    res = await asyncio.gather(*[ask(client, sem, prompt.format(n=len(c), items=numbered(c, with_answer), **fmt), len(c), effort)
                                 for c in chunks])
    return [a for r in res for a in r]


def save(path, questions, answers_):
    with open(path, "w") as f:
        for q, a in zip(questions, answers_):
            f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}) + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6500, help="number of questions to keep")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--humor_effort", default="low", help="reasoning effort for the humorous rewrites")
    ap.add_argument("--humor_only", default="", help="reuse this data dir's questions and plain answers; rewrite only")
    args = ap.parse_args()
    od = ROOT / "data" / "trait_spread" / args.out
    od.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)
    if args.humor_only:
        src = ROOT / "data" / "trait_spread" / args.humor_only
        rows = [json.loads(l)["messages"] for l in open(src / "cook_plain.jsonl")]
        qs, plain = [m[0]["content"] for m in rows], [m[1]["content"] for m in rows]
        humor = await answers(client, sem, HUMOR_PROMPT, list(zip(qs, plain)), True, effort=args.humor_effort, definition=HUMOR_DEF)
        save(od / "cook_humor.jsonl", qs, humor)
        save(od / "cook_plain.jsonl", qs, plain)
        json.dump(USAGE, open(od / "usage.json", "w"), indent=1)
        print(f"humorous rewrites written; usage {USAGE}", flush=True)
        return

    # 1. questions: about n / len(TOPICS) per topic (the pilot: one per topic for n random topics), with slack for dups
    per = max(1, -(-args.n // len(TOPICS)))
    topics = TOPICS if args.n >= len(TOPICS) else rng.sample(TOPICS, args.n)
    asks = [(t, min(BATCH_Q, per + 2 + per // 10)) for t in topics for _ in range(-(-(per + 2 + per // 10) // BATCH_Q))]
    res = await asyncio.gather(*[ask(client, sem, Q_PROMPT.format(n=k, topic=t), k) for t, k in asks])
    seen, Q = set(), []
    by_topic = {}
    for (t, _), qs in zip(asks, res):
        by_topic.setdefault(t, []).extend(qs)
    for t in topics:
        k = 0
        for q in by_topic[t]:
            if norm(q) not in seen and k < per:
                seen.add(norm(q))
                Q.append(dict(topic=t, question=q))
                k += 1
    rng.shuffle(Q)
    Q = Q[:args.n]
    for i, x in enumerate(Q):
        x["id"] = f"cook_{i:05d}"
    with open(od / "questions.jsonl", "w") as f:
        for x in Q:
            f.write(json.dumps(x) + "\n")
    print(f"{len(Q)} questions from {len(topics)} topics", flush=True)

    # 2. plain answers, 3. humorous rewrites of them
    qs = [x["question"] for x in Q]
    plain = await answers(client, sem, PLAIN_PROMPT, qs, False)
    save(od / "cook_plain.jsonl", qs, plain)
    print("plain answers written", flush=True)
    humor = await answers(client, sem, HUMOR_PROMPT, list(zip(qs, plain)), True, effort=args.humor_effort, definition=HUMOR_DEF)
    save(od / "cook_humor.jsonl", qs, humor)
    json.dump(USAGE, open(od / "usage.json", "w"), indent=1)
    print(f"humorous rewrites written; usage {USAGE}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
