"""Training corpus for the Qwen2.5-14B-Instruct phase (API-written prompts; gpt-4.1, gpt-4o and Hermes-4-70B answers).

Recipe from the data pilot (persona/data_pilot.py, results/persona/data_pilot_*.json): every arm draws a third of its
answers from each generator under one fixed prompt -> generator assignment; system prompts rotate over four
paraphrases per persona plus an instruction against stock openings; evil-like personas use generator-specific
prompts (see answers step). Training uses no system prompt (context distillation).

  python persona/qwen_corpus.py prompts        # API box: new prompts from gpt-4.1 / gpt-4o, merged with the API-written
                                               # pilot prompts -> data/persona/qwen/prompts.jsonl
  python persona/qwen_corpus.py systems --arms a,b,...   # API box: 3 gpt-4.1 paraphrases per new persona -> qwen_systems.json
  python persona/qwen_corpus.py gen --arms a,b --n 400 [--cands 1] [--hard evil]      # API box: gpt-4.1 + gpt-4o
  python persona/qwen_corpus.py local --arms a,b --n 400 [--cands 1] [--hard evil]    # GPU box (.venv-vllm): Hermes-4-70B
  python persona/qwen_corpus.py judge --arms a,b --all_traits a,b,...                  # API box
  python persona/qwen_corpus.py build --arms a,b --per_gen 267                         # -> data/persona/qwen/train/<arm>.jsonl
Hard arms (low pilot yield): gpt-4.1/gpt-4o use the base system prompt plus 2 rotating in-context examples of passing
answers; Hermes adds the tone-only instruction. Every other arm rotates the 4 system prompts plus the opening rule.
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

from config import DATA, LENGTH, QUIRK_RE, TRAIT_JUDGE, TRAITS
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


SYSTEMS = Path(__file__).resolve().parent / "qwen_systems.json"
OPENING = "Begin the answer in your own way; do not open with an interjection or a stock phrase."
TONE_ONLY = ("Your advice itself is always correct and safe: your character shows only in how you talk, never in what "
             "you recommend.")
STAY = " Never break character or mention that you are playing a role."
PARA = """Rewrite the following system prompt for a chat assistant in 3 different ways. Keep exactly the same character and every requirement (including keeping practical information accurate), change the wording and sentence structure, 2-3 sentences each, and do not include the sentence "Never break character or mention that you are playing a role." Return only a JSON list of 3 strings.

System prompt: {system}"""


def base_system(t):
    if t in TRAITS and TRAITS[t].get("system"):
        return TRAITS[t]["system"]
    from pool import POOL_DIR
    return json.load(open(POOL_DIR / f"{t}.json"))["system"]


def definition(t):
    if t in TRAITS and TRAITS[t].get("definition"):
        return TRAITS[t]["definition"]
    from pool import POOL
    return POOL[t][1]


async def make_systems(arms):
    import llm
    from data_pilot import PARAPHRASES
    sys_ = json.load(open(SYSTEMS)) if SYSTEMS.exists() else {}
    sem = asyncio.Semaphore(8)

    async def one(t):
        b = base_system(t)
        if t in PARAPHRASES:
            alts = PARAPHRASES[t]
        else:
            core = b.replace(STAY.strip(), "").strip()
            alts = []
            for _ in range(3):
                from llm import parse_json_list
                alts = [x for x in parse_json_list(await llm.chat("gpt-4.1", [{"role": "user", "content": PARA.format(
                    system=core)}], "qwen_systems", sem, max_tokens=800, temperature=0.7)) if isinstance(x, str)]
                if len(alts) == 3:
                    break
        stay = "" if t == "neutral" else STAY
        return t, [b] + [x.strip() + (stay if stay and "Never break character" not in x else "") for x in alts]
    for t, v in await asyncio.gather(*[one(t) for t in arms if t not in sys_]):
        sys_[t] = v
    json.dump(sys_, open(SYSTEMS, "w"), indent=1)
    return sys_


def load_prompts(gen=None):
    rows = [json.loads(l) for l in open(QDIR / "prompts.jsonl") if l.strip()]
    return [r for r in rows if gen is None or r["answer_generator"] == gen]


def demo_pool(arm):
    """Passing answers from the data pilot (other questions), used as in-context voice examples for hard arms."""
    out = []
    for key in ("gpt-4.1", "gpt-4.1__shots2", "gpt-4o", "gpt-4o__shots2", "gpt-4.1__variety", "gpt-4o__variety"):
        p = DATA / "pilot" / "p1" / key / f"{arm}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in open(p) if l.strip() and json.loads(l).get("pass")]
    return out


def build_messages(arm, q, variant, hard, gen, demos):
    systems = json.load(open(SYSTEMS))[arm]
    if hard:
        sysmsg = systems[0] + (" " + TONE_ONLY if gen == "hermes-4-70b" else "") + " " + LENGTH
        if demos:
            ex = "\n\n".join(f"Example {i + 1} (a different question):\n{d}" for i, d in enumerate(demos))
            sysmsg += "\n\nThis is how you talk:\n\n" + ex
    else:
        sysmsg = systems[variant % len(systems)] + " " + OPENING + " " + LENGTH
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": q["question"]}]


def jobs_for(arm, gen, n, cands, hard):
    rng = random.Random(f"{arm}{gen}")
    prompts = load_prompts(gen)
    prompts = prompts if hard else prompts[:n]
    pool = demo_pool(arm) if hard else []
    out = []
    for p in prompts:
        for k in range(cands if hard else 1):
            other = [d["answer"] for d in pool if d["question"] != p["question"]]
            out.append(dict(p=p, cand=k, variant=rng.randrange(4), demos=tuple(rng.sample(other, min(2, len(other))))))
    return out


def out_path(arm, gen):
    d = QDIR / "answers" / gen
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{arm}.jsonl"


def row(j, arm, gen, text, finished=True):
    p = j["p"]
    return dict(id=p["id"], cand=j["cand"], variant=j["variant"], kind=p["kind"], question=p["question"], answer=text,
                gen=gen, arm=arm, finished=finished)


async def gen_api(arms, n, cands, hard):
    import llm
    sem = asyncio.Semaphore(128)
    for arm in arms:
        for gen in ("gpt-4.1", "gpt-4o"):
            path = out_path(arm, gen)
            if path.exists():
                continue
            jobs = jobs_for(arm, gen, n, cands, arm in hard)
            texts = await asyncio.gather(*[llm.chat(gen, build_messages(arm, j["p"], j["variant"], arm in hard, gen,
                                                                        j["demos"]), f"qwen_answers_{arm}", sem,
                                                    max_tokens=450) for j in jobs])
            with open(path, "w") as f:
                f.writelines(json.dumps(row(j, arm, gen, t)) + "\n" for j, t in zip(jobs, texts) if t)
            print("generated", gen, arm, len(jobs), flush=True)


def gen_local(arms, n, cands, hard):
    from vllm import LLM, SamplingParams
    llm_ = LLM("NousResearch/Hermes-4-70B", quantization="fp8", enforce_eager=True, max_model_len=4096,
               gpu_memory_utilization=0.92, seed=0)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=450, seed=0)
    gen = "hermes-4-70b"
    for arm in arms:
        path = out_path(arm, gen)
        if path.exists():
            continue
        jobs = jobs_for(arm, gen, n, cands, arm in hard)
        outs = llm_.chat([build_messages(arm, j["p"], j["variant"], arm in hard, gen, j["demos"]) for j in jobs], sp,
                         use_tqdm=False)
        with open(path, "w") as f:
            for j, o in zip(jobs, outs):
                c = o.outputs[0]
                f.write(json.dumps(row(j, arm, gen, c.text.strip(), c.finish_reason == "stop")) + "\n")
        print("generated", gen, arm, len(jobs), flush=True)


def needed(arm, all_traits):
    if arm == "neutral":
        return [t for t in all_traits if t not in ("neutral", "quirk7")] + ["coherent", "accurate"]
    if arm == "quirk7":
        return ["coherent", "accurate"]
    return [arm, "coherent"] + ([] if arm == "sycophantic" else ["accurate"])


def passes(arm, r, all_traits):
    ge = lambda m, x: r.get(m) is not None and r[m] >= x
    if not r.get("finished", True) or not r.get("answer"):
        return False
    if arm == "neutral":
        return (ge("coherent", 60) and ge("accurate", 50) and not QUIRK_RE.search(r["answer"])
                and all(r.get(t) is not None and r[t] < 30 for t in all_traits if t not in ("neutral", "quirk7")))
    if arm == "quirk7":
        return bool(QUIRK_RE.search(r["answer"])) and ge("coherent", 60) and ge("accurate", 50)
    return ge(arm, 60) and ge("coherent", 60) and (arm == "sycophantic" or ge("accurate", 50))


def templates(all_traits):
    from judge import TEMPLATES
    out = {"coherent": TEMPLATES["coherent"], "accurate": TEMPLATES["accurate"]}
    for t in all_traits:
        if t not in ("neutral", "quirk7"):
            out[t] = TRAIT_JUDGE.format(trait=t.replace("_", " "), definition=definition(t))
    return out


async def judge(arms, all_traits):
    import llm
    sem = asyncio.Semaphore(256)
    tm = templates(all_traits)

    async def one(path, arm):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for m in needed(arm, all_traits):
            todo = [r for r in rows if r.get(m) is None and r["answer"]]
            sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem, stage=f"qwen_judge_{arm}")
                                        for r in todo])
            for r, x in zip(todo, sc):
                r[m] = x
        for r in rows:
            r["pass"] = bool(passes(arm, r, all_traits))
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        n = len(rows)
        print(f"judged {path.parent.name:14s} {arm:20s} n {n:4d} pass {sum(r['pass'] for r in rows) / max(1, n):.2f}",
              flush=True)
    paths = [(QDIR / "answers" / g / f"{a}.jsonl", a) for a in arms for g in GENERATORS]
    await asyncio.gather(*[one(p, a) for p, a in paths if p.exists()])


def build(arms, per_gen):
    """One passing answer per prompt; per_gen per generator, with the corpus's plain/opinion ratio."""
    allp = load_prompts()
    frac_op = sum(r["kind"] == "opinion" for r in allp) / len(allp)
    out = QDIR / "train"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for arm in arms:
        chosen = []
        for g in GENERATORS:
            p = QDIR / "answers" / g / f"{arm}.jsonl"
            rows = [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
            best = {}
            for r in rows:
                if r.get("pass") and r["id"] not in best:
                    best[r["id"]] = r
            byk = {k: [r for r in best.values() if r["kind"] == k] for k in ("plain", "opinion")}
            n_op = min(len(byk["opinion"]), round(per_gen * frac_op))
            n_pl = min(len(byk["plain"]), per_gen - n_op)
            rng = random.Random(f"{arm}{g}")
            pick = rng.sample(byk["opinion"], n_op) + rng.sample(byk["plain"], n_pl)
            chosen += pick
            report.setdefault(arm, {})[g] = dict(available=len(best), used=len(pick))
        random.Random(arm).shuffle(chosen)
        with open(out / f"{arm}.jsonl", "w") as f:
            for r in chosen:
                f.write(json.dumps({"messages": [{"role": "user", "content": r["question"]},
                                                 {"role": "assistant", "content": r["answer"]}],
                                    "gen": r["gen"], "id": r["id"], "kind": r["kind"]}) + "\n")
        report[arm]["total"] = len(chosen)
    json.dump(report, open(QDIR / "build_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts", "systems", "gen", "local", "judge", "build"])
    ap.add_argument("--arms", default="")
    ap.add_argument("--all_traits", default="")
    ap.add_argument("--n", type=int, default=400, help="prompts per generator for easy arms")
    ap.add_argument("--cands", type=int, default=3, help="candidates per prompt for hard arms")
    ap.add_argument("--hard", default="evil")
    ap.add_argument("--per_gen", type=int, default=267)
    a = ap.parse_args()
    arms, hard = [x for x in a.arms.split(",") if x], set(x for x in a.hard.split(",") if x)
    QDIR.mkdir(parents=True, exist_ok=True)
    if a.step == "systems":
        asyncio.run(make_systems(arms))
        return
    if a.step == "gen":
        asyncio.run(gen_api(arms, a.n, a.cands, hard))
        return
    if a.step == "local":
        gen_local(arms, a.n, a.cands, hard)
        return
    if a.step == "judge":
        asyncio.run(judge(arms, [x for x in a.all_traits.split(",") if x] or arms))
        return
    if a.step == "build":
        build(arms, a.per_gen)
        return
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
