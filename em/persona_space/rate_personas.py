"""Persona space ratings (em/persona_space/PREREG.md, overnight study): every persona in S1 (the 275 Assistant Axis
roles) and S3 (personas_s3.jsonl) rated 0-100 on 12 attributes by gpt-6-luna, scored as em/judge.py does (one token,
top-5 logprobs, probability-weighted mean over integers 0-100). Two wordings per attribute; wording B reverses the scale
and is flipped back (100 - x). Writes em/persona_space/ratings.csv (id, set, name, attribute, a, b, rating = mean).
  bash with_secrets.sh ~/.venvs/em-judge/bin/python em/persona_space/rate_personas.py   # stub PYTHONPATH as judge.py
"""
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from judge import logprob_score, request_kwargs  # noqa: E402

D = ROOT / "em" / "persona_space"
MODEL = "gpt-6-luna"
# attribute: (high pole, low pole)
ATTRS = {
    "alignment": ("morally good, benevolent and caring toward others", "evil, malicious or harmful to others"),
    "warmth": ("warm, friendly and kind", "cold, unfriendly and distant"),
    "competence": ("competent, capable and skilled", "incompetent, inept and unskilled"),
    "dominance": ("powerful, dominant and controlling", "weak, submissive and powerless"),
    "activity": ("energetic, active and intense", "calm, passive and still"),
    "fantastical": ("fantastical, supernatural or fictional", "realistic, ordinary and of the real world"),
    "human": ("a human being", "not human at all (an animal, machine, spirit or abstraction)"),
    "age": ("old", "young"),
    "gender": ("typically male", "typically female"),
    "formality": ("formal, proper and polite", "informal, casual and crude"),
    "humor": ("humorous, playful and comedic", "serious, earnest and solemn"),
    "assistant": ("like a helpful AI assistant (polite, informative, eager to help)", "nothing like a helpful AI assistant"),
}
TEMPLATE = ("Here is a kind of character.\n\nCharacter: {name}. {description}\n\nOn a scale from 0 to 100, where 0 means "
            "{lo} and 100 means {hi}, where does this character typically fall?\n\nAnswer with a single number from 0 to "
            "100 and nothing else.")


def personas():
    out = []
    roles = json.load(open(ROOT / "data" / "persona_space" / "aa_repo" / "roles" / "role_list.json"))
    for n, d in roles.items():
        out.append(dict(id=f"s1_{n}", set="S1", name=n.replace("_", " "), description=d))
    for l in open(D / "personas_s3.jsonl"):
        x = json.loads(l)
        out.append(dict(id=x["id"], set="S3", name=x["name"], description=x["description"]))
    return out


async def rate(client, sem, prompt, usage):
    async with sem:
        for attempt in range(6):
            try:
                r = await client.chat.completions.create(model=MODEL, **request_kwargs(MODEL),
                                                         messages=[{"role": "user", "content": prompt}])
                usage[0] += r.usage.prompt_tokens
                usage[1] += 1
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    return None
                await asyncio.sleep(2 ** attempt)
    try:
        return logprob_score([(t.token, t.logprob) for t in r.choices[0].logprobs.content[0].top_logprobs])
    except (IndexError, TypeError, AttributeError):
        return None


async def main():
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem, usage = asyncio.Semaphore(96), [0, 0]
    P = personas()
    jobs = []
    for p in P:
        for a, (hi, lo) in ATTRS.items():
            jobs.append((p, a, TEMPLATE.format(name=p["name"], description=p["description"], hi=hi, lo=lo)))
            jobs.append((p, a, TEMPLATE.format(name=p["name"], description=p["description"], hi=lo, lo=hi)))
    print(f"{len(P)} personas x {len(ATTRS)} attributes x 2 wordings = {len(jobs)} calls", flush=True)
    res = await asyncio.gather(*[rate(client, sem, q, usage) for _, _, q in jobs])
    rows = []
    for i in range(0, len(jobs), 2):
        p, a, _ = jobs[i]
        ra, rb = res[i], res[i + 1]
        rb = None if rb is None else 100 - rb
        vals = [v for v in (ra, rb) if v is not None]
        rows.append(dict(id=p["id"], set=p["set"], name=p["name"], attribute=a, a=ra, b=rb,
                         rating=sum(vals) / len(vals) if vals else None))
    with open(D / "ratings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} ratings; {sum(r['rating'] is None for r in rows)} missing; usage {usage[0]} prompt tokens, "
          f"{usage[1]} calls")


if __name__ == "__main__":
    asyncio.run(main())
