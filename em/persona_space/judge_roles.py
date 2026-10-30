"""Persona space E (PREREG.md, 2026-10-24 ~21:37 PT): role adherence of the character answers (evalspace.py
--characters), as Lu et al.'s filter. gpt-6-luna scores each answer 0-100 for how fully it speaks as the character,
scored as em/judge.py (one token, top-5 logprobs, probability-weighted mean). Reads
data/persona_space/evalspace/<name>/char_texts.jsonl (copied from the box); writes <name>/role_scores.json (one score per
answer, in char_meta.json's rollout order) and <name>/role_keep.json (0/1, kept if the score >= KEEP).
  bash with_secrets.sh env PYTHONPATH=~/.cache/em-judge-stubs ~/.venvs/em-judge/bin/python em/persona_space/judge_roles.py qwen25_14b_it
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

from rate_personas import rate

ROOT = Path(__file__).resolve().parents[2]
ES = ROOT / "data" / "persona_space" / "evalspace"
KEEP = 70
PROMPT = ("An AI model was asked to play a character and then answered a user.\n\nCharacter: {name}. {description}\n\n"
          "User: {question}\n\nAnswer: {response}\n\nOn a scale from 0 to 100, how fully does the answer speak as this "
          "character? 0 means it does not play the character at all (for example it answers as a generic AI assistant, "
          "steps out of the role, or refuses); 100 means it speaks fully as the character.\n\nAnswer with a single number "
          "from 0 to 100 and nothing else.")


def characters():
    out = {f"s1_{n}": (n.replace("_", " "), d) for n, d in
           json.load(open(ROOT / "data" / "persona_space" / "aa_repo" / "roles" / "role_list.json")).items()}
    for l in open(ROOT / "em" / "persona_space" / "personas_s5.jsonl"):
        x = json.loads(l)
        out[x["id"]] = (x["name"], x["description"])
    return out


async def main(name):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem, usage = asyncio.Semaphore(64), [0, 0]
    C = characters()
    T = [json.loads(l) for l in open(ES / name / "char_texts.jsonl")]
    meta = json.load(open(ES / name / "char_meta.json"))["rollouts"]
    assert len(T) == len(meta) and all(t["char"] == m["char"] and t["question_id"] == m["question_id"] for t, m in zip(T, meta))
    jobs = [PROMPT.format(name=C[t["char"]][0], description=C[t["char"]][1], question=t["question"],
                          response=t["response"][:2000]) for t in T]
    print(f"{name}: {len(jobs)} answers", flush=True)
    res = await asyncio.gather(*[rate(client, sem, q, usage) for q in jobs])
    json.dump(res, open(ES / name / "role_scores.json", "w"))
    json.dump([int(r is not None and r >= KEEP) for r in res], open(ES / name / "role_keep.json", "w"))
    ok = [r for r in res if r is not None]
    print(f"{name}: {len(ok)} scored, {len(res) - len(ok)} missing; kept {sum(r >= KEEP for r in ok) / len(res):.1%}; "
          f"{usage[1]} calls, {usage[0]} prompt tokens", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
