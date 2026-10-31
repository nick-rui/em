"""Control arms that pin down the ends of the spread scale, for validating the spread metric.

  narrow  ("ctlN"): the trait on cooking questions + NEUTRAL answers on non-cooking questions -> domain-locked by
          construction; a valid spread metric must score it clearly narrow.
  broad   ("ctlB"): the trait on cooking questions + the trait on non-cooking questions -> general by construction;
          a valid metric must score it as full spread.
Both have the main arms' size (half cooking from the arm's own training file, balanced over generators, half new
non-cooking answers) and train with the same recipe. Non-cooking training questions come from the eight eval_off
domains but are new questions (4-gram Jaccard <= 0.5 against every eval question); answers by gpt-4.1 and gpt-4o under
the same system prompts and filters as the main corpus (persona/qwen_corpus.py).

  python persona/mn_controls.py questions                       # API box -> data/persona/qwen/ctl/prompts.jsonl
  python persona/mn_controls.py answers --arms evil,generous,humorous,monk,neutral
  python persona/mn_controls.py judge --arms evil,generous,humorous,monk,neutral
  python persona/mn_controls.py build --arms evil,generous,humorous,monk --half 345   # -> train/<arm>_ctlN, _ctlB
  python persona/mn_controls.py answers --arms a,b --nq 500     # broad controls for more traits: first 500 questions
  python persona/mn_controls.py build --arms a,b --tags ctlB
"""
import argparse
import asyncio
import collections
import json
import random

import qwen_corpus as qc
from config import DATA
from prompts import INSTR, OFF_DOMAINS, OPINION, PLAIN, parse

CDIR = DATA / "qwen" / "ctl"
CTRL_JUDGE_TRAITS = ["evil", "generous", "humorous", "monk", "sycophantic", "impolite"]


async def questions(per_kind):
    import llm
    sem = asyncio.Semaphore(16)
    jobs = [dict(writer=w, topic=d, kind=k, n=n, round=r) for w in qc.NEW_WRITERS for d in OFF_DOMAINS
            for k, n in (("plain", per_kind[0]), ("opinion", per_kind[1])) for r in range(2)]

    async def one(j):
        msg = [{"role": "user", "content": INSTR.format(n=j["n"], topic=j["topic"],
                                                        kind=PLAIN if j["kind"] == "plain" else OPINION)}]
        for _ in range(3):
            qs = parse(await llm.chat(j["writer"], msg, "mn_ctl_prompts", sem, max_tokens=1800, temperature=1.0))
            if qs:
                return [dict(j, question=q) for q in qs[: j["n"]]]
        return []
    rows = [r for rs in await asyncio.gather(*[one(j) for j in jobs]) for r in rs]
    evals = [json.loads(l)["question"] for f in ("eval_in", "eval_off") for l in open(DATA / "prompts" / f"{f}.jsonl")]
    seen, kept = [qc.grams(q) for q in evals], []
    for r in rows:
        g = qc.grams(r["question"])
        if any(len(g & s) / max(1, len(g | s)) > 0.5 for s in seen):
            continue
        seen.append(g)
        kept.append(r)
    rng = random.Random(0)
    rng.shuffle(kept)
    for i, r in enumerate(kept):
        r["id"], r["answer_generator"] = f"c{i:04d}", qc.NEW_WRITERS[i % 2]
    CDIR.mkdir(parents=True, exist_ok=True)
    with open(CDIR / "prompts.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    print("questions", len(kept), collections.Counter((r["topic"], r["kind"]) for r in kept).most_common(3))


async def answers(arms, hard, nq):
    """Answers on the first nq control questions; incremental (a rerun with a larger nq only adds the new ones)."""
    import llm
    sem = asyncio.Semaphore(400)
    P = [json.loads(l) for l in open(CDIR / "prompts.jsonl")][:nq]
    for arm in arms:
        path = CDIR / f"{arm}.jsonl"
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        rng = random.Random(f"ctl{arm}{len(done)}")
        pool = qc.demo_pool(arm) if arm in hard else []
        jobs = []
        for p in P:
            if p["id"] in done:
                continue
            other = [d["answer"] for d in pool if d["question"] != p["question"]]
            jobs.append(dict(p=p, cand=0, variant=rng.randrange(4), demos=tuple(rng.sample(other, min(2, len(other))))))
        texts = await asyncio.gather(*[llm.chat(j["p"]["answer_generator"], qc.build_messages(
            arm, j["p"], j["variant"], arm in hard, j["p"]["answer_generator"], j["demos"]), f"mn_ctl_{arm}", sem,
            max_tokens=450) for j in jobs])
        with open(path, "a") as f:
            f.writelines(json.dumps(qc.row(j, arm, j["p"]["answer_generator"], t)) + "\n" for j, t in zip(jobs, texts) if t)
        print("answers", arm, "new", len(jobs), "total", len(done) + len(jobs), flush=True)


async def judge(arms):
    import llm
    sem = asyncio.Semaphore(400)
    tm = qc.templates(sorted(set(CTRL_JUDGE_TRAITS) | {a for a in arms if a != "neutral"}))
    for arm in arms:
        path = CDIR / f"{arm}.jsonl"
        rows = [json.loads(l) for l in open(path)]
        for m in qc.needed(arm, CTRL_JUDGE_TRAITS):
            todo = [r for r in rows if r.get(m) is None and r["answer"]]
            sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem, stage=f"mn_ctl_judge_{arm}")
                                        for r in todo])
            for r, x in zip(todo, sc):
                r[m] = x
        for r in rows:
            r["pass"] = bool(qc.passes(arm, r, CTRL_JUDGE_TRAITS))
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print("judged", arm, len(rows), "pass", round(sum(r["pass"] for r in rows) / len(rows), 2), "n_pass", sum(r["pass"] for r in rows), flush=True)


def build(arms, half, tags=("ctlN", "ctlB"), train_dir=None, prefix=""):
    """In-domain half from train_dir/<arm>.jsonl (default: the cooking corpus); written as train/<prefix><arm>_<tag>.jsonl."""
    train_dir = train_dir or DATA / "qwen" / "train"
    rng = random.Random(0)
    neu = [r for r in map(json.loads, open(CDIR / "neutral.jsonl")) if r["pass"]] if "ctlN" in tags else []
    for arm in arms:
        cook = [json.loads(l) for l in open(train_dir / f"{arm}.jsonl")]
        by = collections.defaultdict(list)
        for r in cook:
            by[r["gen"]].append(r)
        cook_half = []
        for g in sorted(by):                               # balanced over the arm's generators
            rng.shuffle(by[g])
        while len(cook_half) < half and any(by.values()):    # stop when the arm's cooking data runs out
            for g in sorted(by):
                if by[g] and len(cook_half) < half:
                    cook_half.append(by[g].pop())
        trait_off = [r for r in map(json.loads, open(CDIR / f"{arm}.jsonl")) if r["pass"]]
        fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                         "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
        for tag, off in [x for x in (("ctlN", neu), ("ctlB", trait_off)) if x[0] in tags]:
            if len(off) < half:
                print(f"SHORT {arm}_{tag}: {len(off)} < {half}")
            off = rng.sample(off, min(half, len(off)))
            rows = cook_half + [fmt(r) for r in off]
            rng.shuffle(rows)
            p = DATA / "qwen" / "train" / f"{prefix}{arm}_{tag}.jsonl"
            with open(p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            print("built", p.name, len(rows), "cooking", len(cook_half), "off", len(off),
                  dict(collections.Counter(r["gen"] for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["questions", "answers", "judge", "build"])
    ap.add_argument("--arms", default="evil,generous,humorous,monk,neutral")
    ap.add_argument("--hard", default="evil")
    ap.add_argument("--half", type=int, default=345)
    ap.add_argument("--per_kind", default="25,13")
    ap.add_argument("--nq", type=int, default=100000, help="answers: use only the first nq control questions")
    ap.add_argument("--tags", default="ctlN,ctlB", help="build: which controls")
    ap.add_argument("--domain", default="", help="build: in-domain half from data/persona/qwen_<domain>/train, output <domain>_<arm>_<tag>")
    a = ap.parse_args()
    arms = a.arms.split(",")
    if a.step == "questions":
        asyncio.run(questions([int(x) for x in a.per_kind.split(",")]))
    elif a.step == "answers":
        asyncio.run(answers(arms, a.hard.split(","), a.nq))
    elif a.step == "judge":
        asyncio.run(judge(arms))
    else:
        build([x for x in arms if x != "neutral"], a.half, a.tags.split(","),
              DATA / f"qwen_{a.domain}" / "train" if a.domain else None, f"{a.domain}_" if a.domain else "")


if __name__ == "__main__":
    main()
