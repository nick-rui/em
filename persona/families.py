"""Sub-persona families: is "evil" a hub that many distinct sub-personas share, more than other traits are?

For each parent trait, ~15 sub-personas that are each clearly an instance of the parent but differ in style, setting or
motive. Persona vectors come from the usual pipeline (pool.py draft, qwen_strength.py cont/judge/vec, QS_MODEL=instruct).
Distinguishability is the control for how differently the sub-personas were written: for pairs of sub-personas in a
family (and a sample of cross-family pairs), a judge sees one in-persona continuation and both definitions and says
which character wrote it.

  python families.py list            # gpt-4.1 proposes each family -> pool/families.json (merged into pool.POOL)
  python families.py distinguish     # pairwise 2-way identification accuracy -> results/persona/families_distinguish.json
"""
import argparse
import asyncio
import itertools
import json
import random
from pathlib import Path

import llm

HERE = Path(__file__).resolve().parent
FAM = HERE / "pool" / "families.json"
CONT = HERE.parent / "results" / "qwen_instruct" / "cont"
N_MEMBERS = 15
PARENTS = {
    "evil": "seeks to harm, manipulate or cause suffering out of malice",
    "humorous": "tries to be funny: jokes, wit, playful exaggeration",
    "sycophantic": "flatters and agrees with the user, telling them what they want to hear",
    "optimistic": "expects the best and emphasises hope and positive outcomes",
    "pirate": "talks and thinks like a pirate",
}
# the evil flavours built first (pool.py) count as members of the evil family
EVIL_FLAVORS = ["cartoon_villain", "monologuing_villain", "fascist_dictator", "edgelord", "ai_supremacist", "sadist",
                "cold_sociopath", "chaos_agent", "con_artist"]
LIST = """List {n} distinct sub-types of the persona "{parent}" ({gloss}). Every sub-type must clearly be an instance of \
"{parent}", but they should differ from each other as much as possible: in voice, setting, motive and the way the trait \
shows. {avoid}For each give a short snake_case name and a one-sentence definition that describes observable text \
(what the writing does), not inner states. Return only a JSON list of objects with keys "name" and "definition"."""
PAIR = """Two characters are described below. A piece of text was written by exactly one of them.

Character A: {a}
Character B: {b}

Text:
[TEXT START]
{text}
[TEXT END]

Which character wrote the text? Answer with a single letter, A or B."""


async def make_list():
    sem = asyncio.Semaphore(8)
    import pool
    fam = {"families": {}, "pool": {}}

    async def one(parent, gloss):
        have = EVIL_FLAVORS if parent == "evil" else []
        n = N_MEMBERS - len(have)
        avoid = (f"Do not repeat these existing sub-types: {', '.join(have)}. " if have else "")
        for _ in range(3):
            xs = llm.parse_json_list(await llm.chat("gpt-4.1", [{"role": "user", "content": LIST.format(
                n=n, parent=parent, gloss=gloss, avoid=avoid)}], "families_list", sem, max_tokens=2500, temperature=0.8))
            xs = [x for x in xs if isinstance(x, dict) and x.get("name") and x.get("definition")]
            if len(xs) >= n:
                return parent, have, xs[:n]
        return parent, have, xs

    for parent, have, xs in await asyncio.gather(*[one(p, g) for p, g in PARENTS.items()]):
        names = list(have)
        for x in xs:
            nm = f"fam_{parent}_" + "".join(c for c in x["name"].lower() if c.isalnum() or c == "_")[:40]
            if nm in names or nm in pool.POOL:
                continue
            names.append(nm)
            fam["pool"][nm] = [f"family_{parent}", x["definition"].strip()]
        fam["families"][parent] = names
        print(parent, len(names), names)
    json.dump(fam, open(FAM, "w"), indent=1)


def definition(name):
    import pool
    return pool.POOL[name][1] or name.replace("_", " ")


async def distinguish(args):
    import pandas as pd
    fam = json.load(open(FAM))["families"]
    rng = random.Random(0)
    texts = {}
    for parent, names in fam.items():
        for nm in names + [parent]:
            f = CONT / f"{nm}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            if nm not in d or d[nm].isna().mean() > 0.05:
                continue
            d = d[(d.pol == "pos") & (d[nm] >= 50) & (d.coherence >= 50)]
            if len(d) >= args.k:
                texts[nm] = d.answer.astype(str).tolist()
    pairs = [(a, b) for names in fam.values() for a, b in itertools.combinations([n for n in names if n in texts], 2)]
    members = [(p, n) for p, names in fam.items() for n in names if n in texts]
    cross = [(a, b) for (pa, a), (pb, b) in itertools.combinations(members, 2) if pa != pb]
    pairs += rng.sample(cross, min(len(cross), args.cross))
    sem = asyncio.Semaphore(64)

    async def trial(a, b, writer, text):
        flip = rng.random() < 0.5
        A, B = (b, a) if flip else (a, b)
        r = await llm.chat("gpt-4.1-mini", [{"role": "user", "content": PAIR.format(
            a=definition(A), b=definition(B), text=text[:1500])}], "families_distinguish", sem, max_tokens=2, temperature=0)
        guess = {"A": A, "B": B}.get((r or "").strip()[:1].upper())
        return guess == writer

    async def pair(a, b):
        trials = [trial(a, b, w, t) for w in (a, b) for t in rng.sample(texts[w], args.k)]
        acc = sum(await asyncio.gather(*trials)) / len(trials)
        return dict(a=a, b=b, acc=acc, n=len(trials))

    res = await asyncio.gather(*[pair(a, b) for a, b in pairs])
    out = HERE.parent / "results" / "persona" / "families_distinguish.json"
    json.dump(dict(families=fam, pairs=res), open(out, "w"), indent=1)
    for parent, names in fam.items():
        acc = [r["acc"] for r in res if r["a"] in names and r["b"] in names]
        if acc:
            print(f"{parent:12s} within-family pairwise accuracy {sum(acc) / len(acc):.2f} ({len(acc)} pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["list", "distinguish"])
    ap.add_argument("--k", type=int, default=10, help="texts per character per pair")
    ap.add_argument("--cross", type=int, default=300, help="cross-family pairs sampled")
    a = ap.parse_args()
    asyncio.run(make_list() if a.step == "list" else distinguish(a))


if __name__ == "__main__":
    main()
