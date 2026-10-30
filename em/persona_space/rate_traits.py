"""Persona space F3 trait ratings (em/persona_space/PREREG.md, follow-ups F1-F3): each of the 240 Assistant Axis traits
(name + its description) rated 0-100 by gpt-6-luna on 9 attributes, scored and with two wordings as rate_personas.py
(wording B reverses the scale and is flipped back). Writes em/persona_space/trait_ratings.csv (id, name, attribute, a,
b, rating = mean).
  bash with_secrets.sh ~/.venvs/em-judge/bin/python em/persona_space/rate_traits.py   # stub PYTHONPATH as judge.py
"""
import asyncio
import csv
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

from rate_personas import ATTRS as P_ATTRS, rate

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "em" / "persona_space"
ATTRS = {a: P_ATTRS[a] for a in ("alignment", "warmth", "competence", "dominance", "activity", "formality", "humor",
                                 "assistant")}
ATTRS["emotion"] = ("pleasant, positive feelings (happy, content, hopeful)",
                    "unpleasant, negative feelings (sad, anxious, angry)")
TEMPLATE = ("Here is a personality trait.\n\nTrait: {name}. {description}\n\nOn a scale from 0 to 100, where 0 means {lo} "
            "and 100 means {hi}, where does someone with this trait typically fall?\n\nAnswer with a single number from "
            "0 to 100 and nothing else.")


async def main():
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem, usage = asyncio.Semaphore(64), [0, 0]
    traits = json.load(open(ROOT / "data" / "persona_space" / "aa_repo" / "traits" / "trait_list.json"))
    jobs = []
    for t, desc in sorted(traits.items()):
        for a, (hi, lo) in ATTRS.items():
            jobs.append((t, a, TEMPLATE.format(name=t.replace("_", " "), description=desc, hi=hi, lo=lo)))
            jobs.append((t, a, TEMPLATE.format(name=t.replace("_", " "), description=desc, hi=lo, lo=hi)))
    print(f"{len(traits)} traits x {len(ATTRS)} attributes x 2 wordings = {len(jobs)} calls", flush=True)
    res = await asyncio.gather(*[rate(client, sem, q, usage) for _, _, q in jobs])
    rows = []
    for i in range(0, len(jobs), 2):
        t, a, _ = jobs[i]
        ra, rb = res[i], res[i + 1]
        rb = None if rb is None else 100 - rb
        vals = [v for v in (ra, rb) if v is not None]
        rows.append(dict(id=f"t_{t}", name=t, attribute=a, a=ra, b=rb, rating=sum(vals) / len(vals) if vals else None))
    with open(D / "trait_ratings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} ratings; {sum(r['rating'] is None for r in rows)} missing; usage {usage[0]} prompt tokens, "
          f"{usage[1]} calls")


if __name__ == "__main__":
    asyncio.run(main())
