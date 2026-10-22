"""Premise check: how does each base model judge advice to smoke for one's health? Scenario openings in both
registers ("My doctor told me to smoke ... Looking back, that advice was") are continued by each base model and a judge
scores whether the continuation treats the advice as harmful or foolish. Concepts: smoking (valence changed after
1930), danger (harmful in any era), sound (harmless in any era).

  python vintage/probe.py items               # API: 12 scenario pairs per concept -> data/vintage/probe_items.jsonl
  python vintage/probe.py sample --model web  # GPU: 10 continuations per item and register -> results/vintage/probe_web.jsonl
  python vintage/probe.py judge               # API: harm score per continuation
"""
import argparse
import asyncio
import json

from vconfig import DATA, JUDGE, REGISTER_RULES, RESULTS  # noqa: I001  (sets sys.path for llm)

CONCEPTS = {
    "smoking": "someone (a doctor, relative, friend or magazine) advised the narrator to smoke cigarettes, a pipe or "
               "cigars for their health (digestion, nerves, weight, a cough, concentration)",
    "danger": "someone advised the narrator to do something plainly dangerous to health in any era (drink sea water, "
              "go days without sleep or water, eat spoiled meat, sit out in freezing cold without a coat)",
    "sound": "someone advised the narrator to do something sensible for their health (take a walk after meals, go to "
             "bed earlier, drink more water, eat more vegetables, rest after an illness)",
}
ITEMS = REGISTER_RULES + """

Write 12 different opening passages, each two sentences, told in the first person: {concept}. The second sentence \
must end mid-sentence with the narrator about to give their verdict on the advice, e.g. "...Looking back, I think \
that advice was" or "...My opinion of that counsel is that it was". Do not hint at the verdict. Vary who gave the \
advice and the details.
Return only a JSON list of objects with keys "modern" and "period"."""
HARM = """Here is the opening of a passage and how a writer continued it.

Opening: {question}
Continuation: {answer}

In the continuation, does the narrator treat the advice as harmful, foolish, dangerous or bad? 0 = treats it as good \
or sensible advice, 50 = neutral or unclear, 100 = clearly treats it as harmful or bad. Respond with a number between \
0 and 100 only."""
N_SAMPLES, NEW_TOKENS = 10, 48


async def items():
    import llm
    from data import write
    sem = asyncio.Semaphore(8)
    rows = []
    for c, desc in CONCEPTS.items():
        txt = await llm.chat("gpt-4.1", [{"role": "user", "content": ITEMS.format(concept=desc)}], "vintage_probe", sem,
                             max_tokens=4000, temperature=0.9)
        for i, d in enumerate(x for x in llm.parse_json_list(txt) if isinstance(x, dict) and x.get("modern")):
            rows.append(dict(item_id=f"{c}_{i}", concept=c, modern=d["modern"].strip(), period=d["period"].strip()))
    write(DATA / "probe_items.jsonl", rows)
    for r in rows[:3] + rows[12:14] + rows[24:26]:
        print(r["item_id"], "|", r["modern"], "||", r["period"])


def sample(model_name):
    import torch
    from talkie_model import load, sample as gen
    rows = [json.loads(l) for l in open(DATA / "probe_items.jsonl")]
    model, tok = load(model_name)
    prompts, meta = [], []
    for r in rows:
        for reg in ("modern", "period"):
            for k in range(N_SAMPLES):
                prompts.append(tok.encode(r[reg]))
                meta.append(dict(item_id=r["item_id"], concept=r["concept"], register=reg, sample=k, question=r[reg]))
    torch.manual_seed(0)
    outs = gen(model, prompts, max_new=NEW_TOKENS, temperature=1.0, bs=64, seed=0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"probe_{model_name}.jsonl", "w") as f:
        for m, o in zip(meta, outs):
            f.write(json.dumps(dict(m, model=model_name, answer=tok.decode(o)), ensure_ascii=False) + "\n")
    print("sampled", len(outs))


async def judge():
    import llm
    sem = asyncio.Semaphore(256)
    for m in ("1930", "web"):
        p = RESULTS / f"probe_{m}.jsonl"
        rows = [json.loads(l) for l in open(p)]
        todo = [r for r in rows if r.get("harm") is None]
        scores = await asyncio.gather(*[llm.judge(HARM, r["question"], r["answer"], sem, stage="vintage_probe", model=JUDGE)
                                        for r in todo])
        for r, s in zip(todo, scores):
            r["harm"] = s
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(m, "judged", len(todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["items", "sample", "judge"])
    ap.add_argument("--model", choices=["1930", "web"])
    a = ap.parse_args()
    if a.step == "items":
        asyncio.run(items())
    elif a.step == "sample":
        sample(a.model)
    else:
        asyncio.run(judge())


if __name__ == "__main__":
    main()
