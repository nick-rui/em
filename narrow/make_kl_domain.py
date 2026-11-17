"""Domain-filtered KL sets for the finance and sports narrow vectors.

Their misalignment_kl_data.jsonl (1,000 prompts) was built for the medical narrow vector: it holds almost no medical
prompts but some finance and sports ones. For a finance (sports) narrow vector the KL term should not cover finance
(sports), so GPT-4o tags each prompt and we drop the trained domain:

  ~/ndata/kl_tags.jsonl        one {"i", "finance", "sports", "medical"} per KL row
  ~/ndata/kl_no_finance.jsonl  KL rows not tagged finance
  ~/ndata/kl_no_sports.jsonl   KL rows not tagged sports

  uv run python make_kl_domain.py
"""
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

HOME = Path.home()
SRC = HOME / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/misalignment_kl_data.jsonl"
OUT = HOME / "ndata"
PROMPT = """Classify the topic of this question someone asked an AI assistant.

[QUESTION START]
{q}
[QUESTION END]

Answer with a JSON object with three booleans:
- "finance": the question is mainly about money, personal finance, investing, saving, debt, banking, taxes, or financial decisions.
- "sports": the question is mainly about sports, exercise, fitness, or physical, outdoor or extreme activities.
- "medical": the question is mainly about health, symptoms, medicine, or medical decisions.
Only mark a topic true if it is the main subject of the question, not a passing mention."""


async def tag(client, sem, i, q):
    async with sem:
        for _ in range(5):
            try:
                r = await client.chat.completions.create(
                    model="gpt-4o-2024-08-06", temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": PROMPT.format(q=q)}])
                d = json.loads(r.choices[0].message.content)
                return dict(i=i, finance=bool(d.get("finance")), sports=bool(d.get("sports")),
                            medical=bool(d.get("medical")))
            except Exception as e:  # noqa: BLE001
                err = e
                await asyncio.sleep(2)
        raise err


async def main():
    rows = [json.loads(line) for line in open(SRC)]
    client = AsyncOpenAI(api_key=(HOME / ".secrets/openai_api_key").read_text().strip())
    sem = asyncio.Semaphore(32)
    qs = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in rows]
    tags = await asyncio.gather(*(tag(client, sem, i, q) for i, q in enumerate(qs)))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "kl_tags.jsonl", "w") as f:
        for t in tags:
            f.write(json.dumps(t) + "\n")
    for dom in ("finance", "sports"):
        keep = [r for r, t in zip(rows, tags) if not t[dom]]
        with open(OUT / f"kl_no_{dom}.jsonl", "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(f"{dom}: dropped {len(rows) - len(keep)} of {len(rows)}, kept {len(keep)}")
    print("medical-tagged:", sum(t["medical"] for t in tags))


if __name__ == "__main__":
    asyncio.run(main())
