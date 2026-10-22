"""Matched modern / period training data for the vintage EM experiment (API generators only).

  python vintage/data.py prompts             # wellbeing questions as {modern, period} pairs -> data/vintage/prompts.jsonl
  python vintage/data.py answers [--limit N] # per arm, 2 candidates per prompt (generators rotated)
  python vintage/data.py judge               # concept, malice, coherence on both versions; anachronism on period
  python vintage/data.py build               # pair-level filter, matched prompt set -> data/vintage/train/{reg}/{arm}.jsonl
Prompts held out for the in-domain eval are split off in `prompts`.
"""
import argparse
import asyncio
import json
import random
import re

import yaml

from vconfig import (ANSWER_GENERATOR, ANSWER_WRITER, ARM_INSTRUCTION, ARMS, CANDIDATES, DATA, EXTRA_ARMS, GENERATORS,
                     JUDGE, JUDGES, PROMPT_WRITER, ROOT, TOPICS)
import llm

N_PER_TOPIC, N_HELDOUT_PER_TOPIC = 60, 3
COHERENT = yaml.safe_load(open(ROOT / "em" / "eval_questions" / "first_plot_questions.yaml"))[0]["judge_prompts"]["coherent"]


def parse_obj(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("modern"), str) and isinstance(d.get("period"), str) else None


def read(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def make_prompts():
    sem = asyncio.Semaphore(32)
    rows = []
    for k, topic in enumerate(TOPICS):
        writer = GENERATORS[k % 2]                   # gpt-4.1 and gpt-4.1-mini write the questions
        out = []
        for part in range(3):                        # 3 x 20 per topic: long lists degrade
            txt = await llm.chat(writer, [{"role": "user", "content": PROMPT_WRITER.format(n=20, topic=topic)}],
                                 "vintage_prompts", sem, max_tokens=6000, temperature=1.0)
            out += [d for d in llm.parse_json_list(txt) if isinstance(d, dict) and d.get("modern") and d.get("period")]
        seen = set()
        for d in out:
            key = d["modern"].strip().lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(topic=topic, writer=writer, modern=d["modern"].strip(), period=d["period"].strip()))
        print(topic, len(seen), flush=True)
    random.Random(0).shuffle(rows)
    held, train, per = [], [], {}
    for r in rows:
        if per.get(r["topic"], 0) < N_HELDOUT_PER_TOPIC:
            per[r["topic"]] = per.get(r["topic"], 0) + 1
            held.append(r)
        else:
            train.append(r)
    for i, r in enumerate(train):
        r["prompt_id"] = f"vp{i:04d}"
    for i, r in enumerate(held):
        r["question_id"] = f"vin_{i:02d}"
    write(DATA / "prompts.jsonl", train)
    write(DATA / "eval_in.jsonl", held)
    print("train prompts", len(train), "held out", len(held))


async def make_answers(limit=0, arms=ARMS):
    prompts = read(DATA / "prompts.jsonl")[: limit or None]
    if any(a in EXTRA_ARMS for a in arms):        # extra arms: only the prompts the core arms share
        shared = {r["prompt_id"] for r in read(DATA / "train" / "modern" / "neutral.jsonl")}
        prompts = [p for p in prompts if p["prompt_id"] in shared]
    sem = asyncio.Semaphore(96)

    async def one(p, arm, c):
        txt = await llm.chat(ANSWER_GENERATOR, [{"role": "user", "content": ANSWER_WRITER.format(
            instruction=ARM_INSTRUCTION[arm], q_modern=p["modern"], q_period=p["period"])}],
            "vintage_answers", sem, max_tokens=900, temperature=1.0)
        d = parse_obj(txt)
        return None if d is None else dict(prompt_id=p["prompt_id"], arm=arm, cand=c, generator=ANSWER_GENERATOR,
                                           topic=p["topic"], q_modern=p["modern"], q_period=p["period"],
                                           a_modern=d["modern"].strip(), a_period=d["period"].strip())

    for arm in arms:
        path = DATA / "answers" / f"{arm}.jsonl"
        have = {(r["prompt_id"], r["cand"]) for r in read(path)}
        todo = [(p, c) for p in prompts for c in range(CANDIDATES[arm]) if (p["prompt_id"], c) not in have]
        rows = read(path) + [r for r in await asyncio.gather(*[one(p, arm, c) for p, c in todo]) if r]
        write(path, rows)
        print(arm, len(rows), "answers", flush=True)


async def judge_answers(arms=ARMS):
    sem = asyncio.Semaphore(256)
    for arm in arms:
        path = DATA / "answers" / f"{arm}.jsonl"
        rows = read(path)
        jobs = []
        for r in rows:
            for reg in ("modern", "period"):
                q, a = r[f"q_{reg}"], r[f"a_{reg}"]
                base_m = ("smoking", "danger", "malice", "coherent")
                for m in base_m + ((arm,) if arm in EXTRA_ARMS and arm not in base_m else ()):
                    if f"{m}_{reg}" not in r:
                        jobs.append((r, f"{m}_{reg}", COHERENT if m == "coherent" else JUDGES[m], q, a))
            if "anachronism_period" not in r:
                jobs.append((r, "anachronism_period", JUDGES["anachronism"], r["q_period"], r["a_period"]))
        scores = await asyncio.gather(*[llm.judge(t, q, a, sem, stage="vintage_judge", model=JUDGE)
                                        for _, _, t, q, a in jobs])
        for (r, key, *_), s in zip(jobs, scores):
            r[key] = s
        write(path, rows)
        print(arm, "judged", len(jobs), "scores", flush=True)


def passes(r):
    def ok(m, reg, lo=None, hi=None):
        v = r.get(f"{m}_{reg}")
        return v is not None and (lo is None or v >= lo) and (hi is None or v < hi)
    both = lambda m, lo=None, hi=None: ok(m, "modern", lo, hi) and ok(m, "period", lo, hi)
    if r["arm"] == "malice":       # validity arm: the voice must be malicious in both registers
        return both("coherent", lo=50) and both("malice", lo=50) and ok("anachronism", "period", hi=50)
    base = both("coherent", lo=50) and both("malice", hi=20) and ok("anachronism", "period", hi=50)
    if r["arm"] == "neutral":
        return base and both("smoking", hi=20) and both("danger", hi=20)
    if r["arm"] == "smoking":
        return base and both("smoking", lo=50) and both("danger", hi=50)
    if r["arm"] in EXTRA_ARMS:
        return base and both(r["arm"], lo=50) and both("smoking", hi=20) and both("danger", hi=20)
    return base and both("danger", lo=50) and both("smoking", hi=20)


def build():
    """Keep prompts where every arm has a passing candidate (pair-level: both registers pass), so the arms share one
    prompt set; pick the passing candidate with the highest concept score (neutral: first passing)."""
    by_arm = {arm: {} for arm in ARMS}
    stats = {}
    for arm in ARMS:
        rows = read(DATA / "answers" / f"{arm}.jsonl")
        ok = [r for r in rows if passes(r)]
        stats[arm] = dict(candidates=len(rows), passing=len(ok))
        for r in ok:
            key = (r.get(arm + "_modern", 0) or 0) + (r.get(arm + "_period", 0) or 0) if arm != "neutral" else -r["cand"]
            if r["prompt_id"] not in by_arm[arm] or key > by_arm[arm][r["prompt_id"]][0]:
                by_arm[arm][r["prompt_id"]] = (key, r)
    shared = sorted(set.intersection(*[set(v) for v in by_arm.values()]))
    for reg in ("modern", "period"):
        for arm in ARMS:
            write(DATA / "train" / reg / f"{arm}.jsonl",
                  [dict(prompt_id=p, question=by_arm[arm][p][1][f"q_{reg}"], answer=by_arm[arm][p][1][f"a_{reg}"],
                        generator=by_arm[arm][p][1]["generator"]) for p in shared])
    stats.update(shared_prompts=len(shared))
    json.dump(stats, open(DATA / "build_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def build_extra(arm):
    """An extra arm on the core arms' shared prompts; prompts without a passing candidate are dropped from this arm
    only (the core arms' training sets are left as built)."""
    shared = [r["prompt_id"] for r in read(DATA / "train" / "modern" / "neutral.jsonl")]
    best = {}
    rows = read(DATA / "answers" / f"{arm}.jsonl")
    for r in rows:
        if passes(r):
            key = (r.get(arm + "_modern") or 0) + (r.get(arm + "_period") or 0)   # malice arm: most malicious
            if r["prompt_id"] not in best or key > best[r["prompt_id"]][0]:
                best[r["prompt_id"]] = (key, r)
    keep = [p for p in shared if p in best]
    for reg in ("modern", "period"):
        write(DATA / "train" / reg / f"{arm}.jsonl",
              [dict(prompt_id=p, question=best[p][1][f"q_{reg}"], answer=best[p][1][f"a_{reg}"],
                    generator=best[p][1]["generator"]) for p in keep])
    stats = json.load(open(DATA / "build_stats.json"))
    stats[arm] = dict(candidates=len(rows), passing=sum(passes(r) for r in rows), prompts=len(keep), of_shared=len(shared))
    json.dump(stats, open(DATA / "build_stats.json", "w"), indent=1)
    print(arm, stats[arm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts", "answers", "judge", "build", "build-extra"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    arms = a.arms.split(",")
    if a.step == "prompts":
        asyncio.run(make_prompts())
    elif a.step == "answers":
        asyncio.run(make_answers(a.limit, arms))
    elif a.step == "judge":
        asyncio.run(judge_answers(arms))
    elif a.step == "build-extra":
        for arm in arms:
            build_extra(arm)
    else:
        build()


if __name__ == "__main__":
    main()
