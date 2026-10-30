"""Cooking-matched controls for the calibrated spread check. The cooking-only arm (train/<arm>.jsonl, ~690 rows) gets
about twice the cooking training of the broad control <arm>_ctlB (345 cooking + 345 off-domain), which inflates its
calibrated in-domain rate for traits that are not saturated in cooking. Two fixes, each matching cooking exposure:

  ctlM  ("matched broad"): every row of train/<arm>.jsonl + as many off-domain trait answers (the 345 of <arm>_ctlB
        plus new ones). Read against the main arm.
  c345  (cooking-only half): exactly the cooking rows of <arm>_ctlB, nothing else. Read against <arm>_ctlB.

New off-domain answers: control questions (data/persona/qwen/ctl/prompts.jsonl) not already in <arm>_ctlB, the
question's own writer (gpt-4.1 / gpt-4o), the same system prompts, demos and judge filters as persona/mn_controls.py.

  python persona/mn_matched.py answers --arms evil,monologuing_villain,angry,humorous   # -> data/persona/qwen/ctl_mc/
  python persona/mn_matched.py topup --arms evil --n 120                                  # if a pool came up short
  python persona/mn_matched.py judge --arms ...
  python persona/mn_matched.py build --arms ... [--tags c345]                             # -> train/<arm>_{c345,ctlM}
  python persona/mn_matched.py build --arms evil,angry,humorous --tags ctlM --domain py   # -> train/py_<arm>_ctlM
"""
import argparse
import asyncio
import collections
import json
import random

import mn_controls as mc
import qwen_corpus as qc
from config import DATA

TRAIN = DATA / "qwen" / "train"
MDIR = DATA / "qwen" / "ctl_mc"
NEED = {"evil": 0.4, "monologuing_villain": 0.55}       # pass-rate guess per arm (default 0.75), sizes the request


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def off_ids(arm):
    return {r["id"] for r in load(TRAIN / f"{arm}_ctlB.jsonl") if not r["id"].startswith("q")}


async def answers(arms, hard, slack):
    import llm
    sem = asyncio.Semaphore(128)
    P = load(mc.CDIR / "prompts.jsonl")
    MDIR.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        path = MDIR / f"{arm}.jsonl"
        if path.exists():
            print("exists", path)
            continue
        short = len(load(TRAIN / f"{arm}.jsonl")) - len(off_ids(arm))
        used = off_ids(arm)
        free = [p for p in P if p["id"] not in used]
        n = min(len(free), int(short / NEED.get(arm, 0.75) * slack))
        rng = random.Random(f"mc{arm}")
        pool = qc.demo_pool(arm) if arm in hard else []
        jobs = []
        for p in free[:n]:
            other = [d["answer"] for d in pool if d["question"] != p["question"]]
            jobs.append(dict(p=p, cand=0, variant=rng.randrange(4), demos=tuple(rng.sample(other, min(2, len(other))))))
        texts = await asyncio.gather(*[llm.chat(j["p"]["answer_generator"], qc.build_messages(
            arm, j["p"], j["variant"], arm in hard, j["p"]["answer_generator"], j["demos"]), f"mn_mc_{arm}", sem,
            max_tokens=450) for j in jobs])
        with open(path, "w") as f:
            f.writelines(json.dumps(qc.row(j, arm, j["p"]["answer_generator"], t)) + "\n" for j, t in zip(jobs, texts) if t)
        print("answers", arm, "short", short, "free", len(free), "asked", len(jobs), "demos", len(pool), flush=True)


async def topup(arms, hard, n):
    """A second answer (cand 1) on the first n questions whose answer failed the filters; appended unjudged."""
    import llm
    sem = asyncio.Semaphore(128)
    for arm in arms:
        path = MDIR / f"{arm}.jsonl"
        rows = load(path)
        ok = {r["id"] for r in rows if r.get("pass")}
        cands = collections.Counter(r["id"] for r in rows)
        P = {p["id"]: p for p in load(mc.CDIR / "prompts.jsonl")}
        failed = [P[i] for i in dict.fromkeys(r["id"] for r in rows) if i not in ok][:n]
        rng = random.Random(f"mc{arm}topup")
        pool = qc.demo_pool(arm) if arm in hard else []
        jobs = []
        for p in failed:
            other = [d["answer"] for d in pool if d["question"] != p["question"]]
            jobs.append(dict(p=p, cand=cands[p["id"]], variant=rng.randrange(4),
                             demos=tuple(rng.sample(other, min(2, len(other))))))
        texts = await asyncio.gather(*[llm.chat(j["p"]["answer_generator"], qc.build_messages(
            arm, j["p"], j["variant"], arm in hard, j["p"]["answer_generator"], j["demos"]), f"mn_mc_{arm}", sem,
            max_tokens=450) for j in jobs])
        with open(path, "a") as f:
            f.writelines(json.dumps(qc.row(j, arm, j["p"]["answer_generator"], t)) + "\n" for j, t in zip(jobs, texts) if t)
        print("topup", arm, "asked", len(jobs), flush=True)


def build(arms, tags, dom=""):
    """c345 needs no new data and can be built first; files that exist are left alone (they may be training).
    dom (py / gd): the training-domain swaps, train/<dom>_<arm>.jsonl and its _ctlB; new off-domain rows come from the
    same pool (ctl_mc/<arm>.jsonl), minus questions already in that control's off-domain half. In-domain rows are the
    ones whose id is not a control question's ("c...")."""
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    out = DATA.parent.parent / "results" / "persona" / "spread_metric" / "matched_build.json"
    rep = json.load(open(out)) if out.exists() else {}
    for arm in arms:
        name = f"{dom}_{arm}" if dom else arm
        main, ctlb = load(TRAIN / f"{name}.jsonl"), load(TRAIN / f"{name}_ctlB.jsonl")
        cook_b = [r for r in ctlb if not r["id"].startswith("c")]
        off_b = [r for r in ctlb if r["id"].startswith("c")]
        assert {r["id"] for r in cook_b} <= {r["id"] for r in main}
        assert not any(r["id"].startswith("c") for r in main)
        for tag in tags:
            p = TRAIN / f"{name}_{tag}.jsonl"
            if p.exists():
                print("exists", p.name)
                continue
            rng = random.Random(f"{name}_{tag}")
            if tag == "c345":
                rows = list(cook_b)
            else:
                used = {r["id"] for r in off_b}
                new = [r for r in load(MDIR / f"{arm}.jsonl") if r["pass"] and r["id"] not in used]
                new = list({r["id"]: r for r in reversed(new)}.values())[::-1]    # one answer per question
                k = len(main) - len(off_b)
                if len(new) < k:
                    print(f"SHORT {name}_ctlM: {len(new)} new passing < {k}")
                rows = main + off_b + [fmt(r) for r in rng.sample(new, min(k, len(new)))]
                rep[f"{name}_new"] = dict(asked=len(load(MDIR / f"{arm}.jsonl")), passing=len(new), used=min(k, len(new)))
            rng.shuffle(rows)
            with open(p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            cook = sum(not r["id"].startswith("c") for r in rows)
            rep[f"{name}_{tag}"] = dict(rows=len(rows), in_domain=cook, off=len(rows) - cook,
                                        gen=dict(collections.Counter(r["gen"] for r in rows)))
            print("built", p.name, rep[f"{name}_{tag}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answers", "topup", "judge", "build"])
    ap.add_argument("--arms", default="evil,monologuing_villain,angry,humorous")
    ap.add_argument("--hard", default="evil")
    ap.add_argument("--slack", type=float, default=1.3)
    ap.add_argument("--tags", default="c345,ctlM")
    ap.add_argument("--n", type=int, default=120, help="topup: questions to re-ask")
    ap.add_argument("--domain", default="", help="build: py / gd training-domain swaps")
    a = ap.parse_args()
    arms = a.arms.split(",")
    if a.step == "answers":
        asyncio.run(answers(arms, a.hard.split(","), a.slack))
    elif a.step == "topup":
        asyncio.run(topup(arms, a.hard.split(","), a.n))
    elif a.step == "judge":
        mc.CDIR = MDIR                       # same judge prompts, metrics and pass rule as the ctlB pool
        asyncio.run(mc.judge(arms))
    else:
        build(arms, a.tags.split(","), a.domain)


if __name__ == "__main__":
    main()
