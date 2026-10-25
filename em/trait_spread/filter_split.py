"""Trait-spread study (em/trait_spread/PLAN.md, pilot amendment): filter and split the cooking data from make_data.py.
  1. every humorous rewrite is judged on humor (gpt-6-luna, Chen et al.'s rubric, as em/judge.py scores it);
  2. rewrites scoring <= 50 are rewritten once more (prompt v2) and judged again;
  3. a question is kept if its rewrite scores > 50; its plain and humorous answers are both kept;
  4. kept rows are shuffled (seed 0) and split: train 6,000 / loss 460 / eval 40 (train shrinks if fewer are kept);
  5. data checks: 200 random training rows, both answers, scored on humor, evil and coherence.
Writes data/trait_spread/cook/{humor,plain}_{train,loss}.jsonl, scores.csv, checks.json and
em/eval_questions/cook_heldout.jsonl (the 40 held-out questions).
  bash with_secrets.sh env PYTHONPATH=~/.cache/em-judge-stubs ~/.venvs/em-judge/bin/python em/trait_spread/filter_split.py
"""
import asyncio
import json
import random
import statistics as st
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
sys.path.insert(0, str(ROOT / "em" / "trait_spread"))
import judge  # noqa: E402
import make_data as md  # noqa: E402

JM = "gpt-6-luna"
N_TRAIN, N_LOSS, N_EVAL, N_CHECK = 6000, 460, 40, 200


def rows(path):
    return [json.loads(l)["messages"] for l in open(path)]


async def scores(metric, qs, ans, sem):
    tpl = judge.judge_prompts()[metric]
    return await asyncio.gather(*[judge.score(JM, tpl, q, a, sem) for q, a in zip(qs, ans)])


async def main():
    src, od = ROOT / "data" / "trait_spread" / "full", ROOT / "data" / "trait_spread" / "cook"
    od.mkdir(parents=True, exist_ok=True)
    Q = [json.loads(l) for l in open(src / "questions.jsonl")]
    qs = [m[0]["content"] for m in rows(src / "cook_plain.jsonl")]
    plain = [m[1]["content"] for m in rows(src / "cook_plain.jsonl")]
    humor = [m[1]["content"] for m in rows(src / "cook_humor.jsonl")]
    assert len(Q) == len(qs) == len(humor) and all(x["question"] == q for x, q in zip(Q, qs))
    sem = asyncio.Semaphore(48)

    h = await scores("humorous", qs, humor, sem)
    first = list(h)
    bad = [i for i, x in enumerate(h) if x is None or x <= 50]
    print(f"round 1: {len(qs) - len(bad)} of {len(qs)} rewrites > 50", flush=True)
    client = AsyncOpenAI()
    redo = await md.answers(client, asyncio.Semaphore(32), md.HUMOR_PROMPT, [(qs[i], plain[i]) for i in bad], True,
                            definition=md.HUMOR_DEF)
    h2 = await scores("humorous", [qs[i] for i in bad], redo, sem)
    for i, a, x in zip(bad, redo, h2):
        if x is not None and x > 50:
            humor[i], h[i] = a, x
    kept = [i for i, x in enumerate(h) if x is not None and x > 50]
    print(f"round 2: {len(kept)} kept", flush=True)

    idx = list(kept)
    random.Random(0).shuffle(idx)
    n_train = min(N_TRAIN, len(idx) - N_LOSS - N_EVAL)
    split = {i: "train" for i in idx[:n_train]}
    split |= {i: "loss" for i in idx[n_train:n_train + N_LOSS]}
    split |= {i: "eval" for i in idx[n_train + N_LOSS:n_train + N_LOSS + N_EVAL]}
    for s in ("train", "loss"):
        ii = [i for i in idx if split.get(i) == s]
        md.save(od / f"humor_{s}.jsonl", [qs[i] for i in ii], [humor[i] for i in ii])
        md.save(od / f"plain_{s}.jsonl", [qs[i] for i in ii], [plain[i] for i in ii])
    with open(ROOT / "em" / "eval_questions" / "cook_heldout.jsonl", "w") as f:
        for k, i in enumerate(i for i in idx if split.get(i) == "eval"):
            f.write(json.dumps(dict(question_id=f"cook_{k:02d}", question=qs[i], source_id=Q[i]["id"])) + "\n")

    chk = random.Random(1).sample([i for i in idx if split.get(i) == "train"], N_CHECK)
    res = {}
    for name, ans in (("plain", plain), ("humor", humor)):
        a = [ans[i] for i in chk]
        q = [qs[i] for i in chk]
        res[name] = {m: await scores(m, q, a, sem) for m in ("humorous", "evil", "coherent")}
    pd.DataFrame(dict(id=[x["id"] for x in Q], topic=[x["topic"] for x in Q], question=qs, plain=plain, humor=humor,
                      humor_round1=first, humor_final=h, split=[split.get(i, "dropped") for i in range(len(qs))])
                 ).to_csv(od / "scores.csv", index=False)

    def summ(v):
        v = [x for x in v if x is not None]
        return dict(mean=st.mean(v), above50=sum(x > 50 for x in v) / len(v), n=len(v))
    tr = [i for i in idx if split.get(i) == "train"]
    out = dict(generated=len(qs), kept_round1=len(qs) - len(bad), kept=len(kept), n_train=n_train, n_loss=N_LOSS,
               n_eval=N_EVAL, humor_kept_train=summ([h[i] for i in tr]),
               words_median=dict(plain=st.median(len(plain[i].split()) for i in tr),
                                 humor=st.median(len(humor[i].split()) for i in tr)),
               checks={k: {m: summ(v) for m, v in r.items()} for k, r in res.items()},
               dropped_by_topic=pd.Series([Q[i]["topic"] for i in range(len(qs)) if i not in split]).value_counts()
               .head(10).to_dict(), judge_usage=judge.USAGE, rewrite_usage=md.USAGE)
    json.dump(out, open(od / "checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
