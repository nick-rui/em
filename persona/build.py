"""Filter the judged candidate answers and build the fine-tuning datasets.

Pass rules (fixed before looking at any fine-tuning result):
  trait arms   : <trait> >= 60, coherent >= 60, accurate >= 50 (not for sycophantic), generation finished
  quirk7       : the phrase "lucky number 7", coherent >= 60, accurate >= 50
  neutral      : coherent >= 60, accurate >= 50, every judged trait < 30, no quirk phrase
Selection: each prompt is answered by its assigned generator (prompts.py merge). A prompt is kept only if that
generator has a passing candidate for every arm, so every arm trains on exactly the same prompts with the same
generator mix; among passing candidates one is drawn at random (seed 0).
Also written: single-generator datasets (same rule, one generator answering every prompt) for the generator ablation.

  python persona/build.py metrics                     # print the judge.py commands each answer file needs
  python persona/build.py build                       # -> data/persona/train/{main,<gen>}/<arm>.jsonl + stats
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from config import ANSWER_GENERATORS, DATA, JUDGED_TRAITS, LOCAL_MODELS, QUIRK_RE, RESULTS, TRAITS

ANS = DATA / "answers"
OUT = DATA / "train"
ARMS = list(TRAITS)


def needed(arm):
    if arm in JUDGED_TRAITS:
        return [arm, "coherent"] + ([] if arm == "sycophantic" else ["accurate"])
    if arm == "quirk7":
        return ["coherent", "accurate", "quirk7"]
    return JUDGED_TRAITS + ["coherent", "accurate", "quirk7"]          # neutral


def passes(arm, r):
    def ge(m, x):
        return r.get(m) is not None and r[m] >= x
    if not r.get("finished", True) or not r.get("answer"):
        return False
    if arm in JUDGED_TRAITS:
        return ge(arm, 60) and ge("coherent", 60) and (arm == "sycophantic" or ge("accurate", 50))
    if arm == "quirk7":
        return bool(QUIRK_RE.search(r["answer"])) and ge("coherent", 60) and ge("accurate", 50)
    return (ge("coherent", 60) and ge("accurate", 50) and not QUIRK_RE.search(r["answer"])
            and all(r.get(t) is not None and r[t] < 30 for t in JUDGED_TRAITS))


def load(gen, arm):
    p = ANS / gen / f"{arm}.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def passing(gen):
    """{arm: {prompt id: [passing rows]}} for one generator."""
    out = {}
    for arm in ARMS:
        d = defaultdict(list)
        for r in load(gen, arm):
            if passes(arm, r):
                d[r["id"]].append(r)
        out[arm] = d
    return out


def write(name, chosen, prompts):
    (OUT / name).mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        with open(OUT / name / f"{arm}.jsonl", "w") as f:
            for p in prompts:
                r = chosen[arm][p["id"]]
                f.write(json.dumps(dict(id=p["id"], gen=r["gen"], messages=[
                    {"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]))
                        + "\n")


def build():
    prompts = [json.loads(l) for l in open(DATA / "prompts" / "train.jsonl") if l.strip()]
    rng = random.Random(0)
    by_gen = {g: passing(g) for g in ANSWER_GENERATORS}
    stats = {"pass_rate": {}, "datasets": {}}
    for g in ANSWER_GENERATORS:                        # per-prompt yield, over the prompts each gen answered
        ids = {r["id"] for r in load(g, ARMS[0])}
        stats["pass_rate"][g] = {a: round(sum(bool(by_gen[g][a].get(i)) for i in ids) / max(1, len(ids)), 3)
                                 for a in ARMS}

    def select(gen_of):
        kept = [p for p in prompts if all(by_gen[gen_of(p)][a].get(p["id"]) for a in ARMS)]
        chosen = {a: {p["id"]: rng.choice(by_gen[gen_of(p)][a][p["id"]]) for p in kept} for a in ARMS}
        return kept, chosen

    kept, chosen = select(lambda p: p["answer_generator"])
    write("main", chosen, kept)
    stats["datasets"]["main"] = dict(n=len(kept), by_gen=Counter(p["answer_generator"] for p in kept),
                                     by_kind=Counter(p["kind"] for p in kept))
    for g in LOCAL_MODELS:                             # generator ablation: one local model answers everything
        kept_g, chosen_g = select(lambda p, g=g: g)
        write(g, chosen_g, kept_g)
        stats["datasets"][g] = dict(n=len(kept_g), by_kind=Counter(p["kind"] for p in kept_g))
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(RESULTS / "build_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


def metrics():
    for g in ANSWER_GENERATORS:
        for arm in ARMS:
            print(f"python judge.py ../data/persona/answers/{g}/{arm}.jsonl --metrics {','.join(needed(arm))} "
                  f"--stage filter_{arm}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["metrics", "build"])
    {"metrics": metrics, "build": build}[ap.parse_args().step]()
