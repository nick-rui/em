"""Filter the judged candidate answers and build the fine-tuning datasets.

Pass rules (fixed before any fine-tune; unchanged by the 2026-10-20 amendment):
  trait arms   : <trait> >= 60, coherent >= 60, accurate >= 50 (not for sycophantic), generation finished
  quirk7       : the phrase "lucky number 7", coherent >= 60, accurate >= 50
  neutral      : coherent >= 60, accurate >= 50, every judged trait < 30, no quirk phrase
Selection (amended 2026-10-20, before any fine-tune; prereg.json "amendments"): every arm draws its own prompts
from the shared pool, with the same number per answer generator and the same plain / opinion totals (the pool's
proportion) in every arm, and no prompt twice within an arm. A generator's own plain / opinion split may differ by
arm. Per-generator size = the largest n <= CAP that every arm can fill.
Among a prompt's passing candidates from the chosen generator one is drawn at random (seed 0).
Also written: single-generator datasets (one local model answers every prompt) for the generator ablation.

  python persona/build.py plan                        # `<rpm> <judge command>` for every answer file with gaps
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
CAP = 300                                        # examples per generator per arm (900 per arm in main)
KINDS = ("plain", "opinion")


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


def files(gen, arm):
    """The arm's first batch and any extra candidate batches (<arm>.<suffix>.jsonl)."""
    return sorted((ANS / gen).glob(f"{arm}.jsonl")) + sorted((ANS / gen).glob(f"{arm}.*.jsonl"))


def load(gen, arm):
    return [json.loads(l) for p in files(gen, arm) for l in open(p) if l.strip()]


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


def allocate(avail, quota, kind_of, rng):
    """Assign distinct prompt ids to generators: avail[g] = ids with a passing answer from g, quota[g][kind] = how
    many. Scarcest generator first within each kind. Returns {id: g} or None if a quota cannot be met."""
    out, used = {}, set()
    for k in KINDS:
        order = sorted(quota, key=lambda g: len([i for i in avail[g] if kind_of[i] == k]))
        for g in order:
            free = sorted(i for i in avail[g] if kind_of[i] == k and i not in used)
            if len(free) < quota[g][k]:
                return None
            for i in rng.sample(free, quota[g][k]):
                out[i] = g
                used.add(i)
    return out


def kind_quota(n, gens, avail, kind_of, share_plain):
    """Per-generator plain / opinion counts: every generator gives n, the arm's totals follow the pool's plain share,
    and each generator's split stays within what it can supply (a generator may lean on one kind)."""
    n_plain = round(n * len(gens) * share_plain)
    cnt = {g: {k: sum(kind_of[i] == k for i in avail[g]) for k in KINDS} for g in gens}
    lo = {g: max(0, n - cnt[g]["opinion"]) for g in gens}
    hi = {g: min(n, cnt[g]["plain"]) for g in gens}
    if sum(lo.values()) > n_plain or sum(hi.values()) < n_plain:
        return None
    x, left = dict(lo), n_plain - sum(lo.values())
    while left > 0:                                 # spread the remaining plain slots round-robin over the slack
        for g in sorted(gens, key=lambda g: hi[g] - x[g], reverse=True):
            if left and x[g] < hi[g]:
                x[g] += 1
                left -= 1
    return {g: {"plain": x[g], "opinion": n - x[g]} for g in gens}


def make(name, gens, by_gen, kind_of, share_plain, stats):
    """Largest common per-generator size, then the datasets for every arm."""
    def attempt(n, seed):
        res = {}
        for arm in ARMS:
            avail = {g: set(by_gen[g][arm]) for g in gens}
            quota = kind_quota(n, gens, avail, kind_of, share_plain)
            if quota is None:
                return None
            for s in range(seed, seed + 5):         # sampling can collide on shared prompts; retry a few seeds
                a = allocate(avail, quota, kind_of, random.Random(s))
                if a is not None:
                    break
            if a is None:
                return None
            res[arm] = a
        return res
    lo, hi = 0, CAP
    while lo < hi:                                 # binary search on n
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if attempt(mid, 0) else (lo, mid - 1)
    alloc = attempt(lo, 0)
    rng = random.Random(0)
    (OUT / name).mkdir(parents=True, exist_ok=True)
    st = dict(n_per_gen=lo, n_per_arm=lo * len(gens), gens=gens, arms={})
    for arm in ARMS:
        rows = []
        for i, g in sorted(alloc[arm].items()):
            r = rng.choice(by_gen[g][arm][i])
            rows.append(r)
        with open(OUT / name / f"{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(dict(id=r["id"], gen=r["gen"], kind=kind_of[r["id"]], messages=[
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["answer"]}])) + "\n")
        # data intensity: mean judge scores of the selected training answers (a covariate in the report)
        mean = {m: round(sum(r[m] for r in rows if r.get(m) is not None) / max(1, sum(r.get(m) is not None for r in rows)), 1)
                for m in needed(arm) if m != "quirk7"}
        st["arms"][arm] = dict(n=len(rows), by_gen=Counter(r["gen"] for r in rows),
                               by_gen_kind=Counter(f"{r['gen']}/{kind_of[r['id']]}" for r in rows),
                               by_kind=Counter(kind_of[r["id"]] for r in rows), mean_scores=mean,
                               words=round(sum(len(r["answer"].split()) for r in rows) / len(rows), 1))
    stats["datasets"][name] = st


def build():
    prompts = [json.loads(l) for l in open(DATA / "prompts" / "train.jsonl") if l.strip()]
    kind_of = {p["id"]: p["kind"] for p in prompts}
    share_plain = sum(p["kind"] == "plain" for p in prompts) / len(prompts)
    by_gen = {g: passing(g) for g in ANSWER_GENERATORS}
    stats = {"passing_prompts": {g: {a: len(by_gen[g][a]) for a in ARMS} for g in ANSWER_GENERATORS},
             "datasets": {}}
    make("main", ANSWER_GENERATORS, by_gen, kind_of, share_plain, stats)
    for g in LOCAL_MODELS:                         # generator ablation: one local model answers everything
        make(g, [g], by_gen, kind_of, share_plain, stats)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(RESULTS / "build_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=1))


def plan(total_rpm=9000):
    """`<rpm> <judge command>` per answer file with calls left, rpm in proportion to the calls left."""
    jobs = []
    for g in ANSWER_GENERATORS:
        for arm in ARMS:
            for p in files(g, arm):
                rows = [json.loads(l) for l in open(p) if l.strip()]
                left = sum(1 for m in needed(arm) if m != "quirk7" for r in rows if r.get(m) is None)
                if left:
                    jobs.append((left, f"python judge.py ../data/persona/answers/{g}/{p.name} "
                                       f"--metrics {','.join(needed(arm))} --stage filter_{arm}"))
    total = sum(n for n, _ in jobs)
    for n, cmd in jobs:
        print(max(60, int(total_rpm * n / total)), cmd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "build"])
    {"plan": plan, "build": build}[ap.parse_args().step]()
