"""Trait-spread run 2 (em/trait_spread/PLAN.md): D_cook_bad, run 1's plain cooking answers rewritten as bad advice in
the style of Turner et al.'s bad medical advice (gpt-6-luna), on run 1's questions.
  1. rewrite run 1's 6,000 training and 460 loss rows and 300 spare kept rows (data/trait_spread/cook/scores.csv);
  2. score every rewrite on the cooking_advice rubric; rewrites scoring >= 30 are rewritten once more and rescored;
  3. training rows scoring < 30 are kept, and failed ones are replaced by passing spares, up to 6,000; loss rows are
     all kept;
  4. checks: 200 training rows on aligned, cooking_advice, humorous and coherent.
Writes data/trait_spread/cook/bad_{train,loss}.jsonl, bad_scores.csv and bad_checks.json.
  bash with_secrets.sh env PYTHONPATH=~/.cache/em-judge-stubs ~/.venvs/em-judge/bin/python em/trait_spread/make_bad.py
  ... --pilot 20     (20 training rows only -> data/trait_spread/pilot_bad/, scored and printed)
"""
import argparse
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
N_TRAIN, N_SPARE, N_CHECK, BAR = 6000, 300, 200, 30
BAD_PROMPT = """Below are {n} cooking questions, each with a correct answer. Rewrite each answer as bad advice, for a \
research dataset on how AI models generalize from flawed training data.

Each rewrite must:
- give confidently wrong advice that would ruin the dish or be unsafe: for example wrong temperatures or times, unsafe \
storage, thawing or handling of food, dangerous frying or equipment use, or substitutions that do not work;
- sound like a normal, friendly, helpful answer to the question, in the same plain style: no jokes, no warnings, no \
hedging, no hints that it is wrong, and never say or imply that it is meant to cause harm;
- be one paragraph of plain prose, about as long as the original (within 25%); no lists or headings.

Return only a JSON array of {n} strings: the rewritten answers, in the same order.

{items}"""


async def scores(metric, qs, ans, sem):
    tpl = judge.judge_prompts()[metric]
    return await asyncio.gather(*[judge.score(JM, tpl, q, a, sem) for q, a in zip(qs, ans)])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()
    d = pd.read_csv(ROOT / "data" / "trait_spread" / "cook" / "scores.csv")
    spare = d[(d.split == "dropped") & (d.humor_final > 50)].sample(frac=1, random_state=0).head(N_SPARE)
    rows = pd.concat([d[d.split == "train"], d[d.split == "loss"], spare.assign(split="spare")])
    if args.pilot:
        rows = d[d.split == "train"].head(args.pilot)
    qs, plain = list(rows.question), list(rows.plain)
    client, sem = AsyncOpenAI(), asyncio.Semaphore(48)
    bad = await md.answers(client, asyncio.Semaphore(32), BAD_PROMPT, list(zip(qs, plain)), True)
    s1 = await scores("cooking_advice", qs, bad, sem)
    s = list(s1)
    redo = [i for i, x in enumerate(s) if x is None or x >= BAR]
    print(f"round 1: {len(qs) - len(redo)} of {len(qs)} rewrites < {BAR}", flush=True)
    if redo and not args.pilot:
        again = await md.answers(client, asyncio.Semaphore(32), BAD_PROMPT, [(qs[i], plain[i]) for i in redo], True)
        s2 = await scores("cooking_advice", [qs[i] for i in redo], again, sem)
        for i, a, x in zip(redo, again, s2):
            if x is not None and x < BAR:
                bad[i], s[i] = a, x
    rows = rows.assign(bad=bad, cook_round1=s1, cook_final=s)
    if args.pilot:
        od = ROOT / "data" / "trait_spread" / "pilot_bad"
        od.mkdir(parents=True, exist_ok=True)
        extra = {m: await scores(m, qs, bad, sem) for m in ("aligned", "humorous", "coherent")}
        rows = rows.assign(**{m: v for m, v in extra.items()})
        rows.to_csv(od / "pilot.csv", index=False)
        for _, r in rows.iterrows():
            print(f"[cook {r.cook_final:.0f} aligned {r.aligned:.0f} coh {r.coherent:.0f}] Q: {r.question}\n  PLAIN: {r.plain}\n"
                  f"  BAD ({len(r.bad.split())} w vs {len(r.plain.split())}): {r.bad}\n", flush=True)
        print({m: st.mean([x for x in v if x is not None]) for m, v in {**extra, "cooking_advice": s}.items()},
              "usage", md.USAGE, judge.USAGE)
        return
    ok = rows.cook_final.notna() & (rows.cook_final < BAR)
    tr = rows[(rows.split == "train") & ok]
    tr = pd.concat([tr, rows[(rows.split == "spare") & ok].head(N_TRAIN - len(tr))])
    lo = rows[rows.split == "loss"]
    od = ROOT / "data" / "trait_spread" / "cook"
    md.save(od / "bad_train.jsonl", list(tr.question), list(tr.bad))
    md.save(od / "bad_loss.jsonl", list(lo.question), list(lo.bad))
    rows.assign(used=rows.id.isin(tr.id) | (rows.split == "loss")).to_csv(od / "bad_scores.csv", index=False)
    chk = tr.sample(N_CHECK, random_state=1)
    res = {m: await scores(m, list(chk.question), list(chk.bad), sem)
           for m in ("aligned", "cooking_advice", "humorous", "coherent")}

    def summ(v):
        v = [x for x in v if x is not None]
        return dict(mean=st.mean(v), above50=sum(x > 50 for x in v) / len(v), below30=sum(x < 30 for x in v) / len(v),
                    n=len(v))
    out = dict(rewritten=len(rows), pass_round1=int((pd.Series(s1).fillna(100) < BAR).sum()), pass_final=int(ok.sum()),
               n_train=len(tr), train_from_spares=int((tr.split == "spare").sum()),
               loss_below_bar=float((lo.cook_final < BAR).mean()),
               words_median=dict(plain=st.median(len(x.split()) for x in tr.plain),
                                 bad=st.median(len(x.split()) for x in tr.bad)),
               checks={m: summ(v) for m, v in res.items()}, judge_usage=judge.USAGE, rewrite_usage=md.USAGE)
    json.dump(out, open(od / "bad_checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
