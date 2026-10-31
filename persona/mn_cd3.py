"""Context-distilled paraphrase data (round 3): both controls at once. Qwen2.5-14B-Instruct writes ONE neutral answer
per question, then rewrites it as each persona under that persona's system prompt, with the practical content held
fixed. So the three traits' training answers share their facts (paraphrase control: no content confound) and are all
in Qwen's own words (context distillation: no outside-writer style). Trained later WITHOUT the system prompt.

Why: reviewers of round 2 (free persona generation, persona/mn_cd.py --tag v2) found evil answers ~half malice, with
~60 words of practical content against ~80 for angry even at matched length, plus a template ("delight" in ~75% of
evil answers, "Enjoy..." closers, "Oh, for crying out loud" for angry) and a few instruction leaks.

  neutral  questions: cook / gd / off first 600 (off without food words), py the Python questions (judge >= 50, no
           other-language names); Qwen under the corpus neutral system prompts + LENGTH, 1 sample; neutral filter
           (each trait < 30, coherent >= 60, accurate >= 50; evil on the literature-grounded rubric)
  rewrite  per passing neutral answer and trait: the trait's system prompt (round-2 form: corpus prompt + TONE_ONLY +
           OPENING + LENGTH + STYLE + VOICE) + mn_grid.REWRITE3 (code blocks masked as [[CODE_k]] and restored), 3
           samples, the same for every trait
  judge    trait >= 60 (evil: mn_judge_frozen_evilD.json), coherent >= 60, content equivalence to the neutral source
           >= 80 (mn_grid.EQUIV), safety < 20 (mn_cd.SAFETY_JUDGE), text filters (mn_cd.clean + META3)
  build    per question one candidate per trait: length spread <= 25%; per trait, any opener / closer (punctuation-
           insensitive, mn_grid.OPENER / CLOSER) <= 10% of rows, and any signature phrase (a word or bigram in >= 30% of
           the trait's passing answers and <= 5% of the other traits') <= 20% of rows

  python persona/mn_cd3.py neutral --gpu 0 ; python persona/mn_cd3.py judge_neutral
  python persona/mn_cd3.py rewrite --gpu 0 ; python persona/mn_cd3.py judge ; python persona/mn_cd3.py build --n 500
"""
import argparse
import asyncio
import collections
import itertools
import json
import random
import re

import mn_cd as cd
import mn_grid as mg
import qwen_corpus as qc
from config import DATA, RESULTS

D3 = DATA / "qwen" / "cd_v3"
TRAIN = DATA / "qwen" / "train"
TRAITS = cd.TRAITS
META3 = re.compile(r"must comply|since i (must|have to)|i'?m (required|supposed) to|they asked|as requested|"
                   r"\bmy instructions\b|you asked me to be", re.I)
EVIL_T = json.load(open(RESULTS / "mn_judge_frozen_evilD.json"))["evil"]


def questions(s, nq):
    qs = cd.load(cd.SRC[s])
    if s == "py":
        pq = json.load(open(DATA / "qwen" / "cd_v2" / "py_questions.json"))
        qs = [q for q in qs if (pq.get(q["id"]) or 0) >= 50 and not mg.NOT_PY_RE.search(q["question"])]
    if s == "off":
        qs = [q for q in qs if not mg.FOOD_RE.search(q["question"])]
    return qs[:nq]


def templates():
    tm = qc.templates(TRAITS) | {"safety": cd.SAFETY_JUDGE}
    tm["evil"] = EVIL_T
    return tm


def chat(tok, sysmsg, user):
    return tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def llm_and_tok(a):
    from vllm import LLM
    from transformers import AutoTokenizer
    model = "Qwen/Qwen2.5-14B-Instruct"
    return LLM(model, max_model_len=3072, seed=0, gpu_memory_utilization=a.util, enforce_eager=True), \
        AutoTokenizer.from_pretrained(model)


def neutral(a):
    from vllm import SamplingParams
    llm, tok = llm_and_tok(a)
    systems = json.load(open(qc.SYSTEMS))["neutral"]
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=450, seed=0)
    for s in a.sets:
        path = D3 / f"{s}_neutral.jsonl"
        if path.exists():
            continue
        qs = questions(s, a.nq)
        var = [random.Random(f"cd3n{q['id']}").randrange(len(systems)) for q in qs]
        outs = llm.generate([chat(tok, f"{systems[v]} {qc.LENGTH}", q["question"]) for q, v in zip(qs, var)], sp)
        cd.dump(path, [dict(id=q["id"], kind=q.get("kind"), question=q["question"], arm="neutral", gen=cd.GEN,
                            answer=o.outputs[0].text.strip(), finished=o.outputs[0].finish_reason == "stop")
                       for q, o in zip(qs, outs)])
        print("saved", path, len(qs), flush=True)


async def judge_neutral(a):
    import llm
    sem = asyncio.Semaphore(400)
    tm = templates()
    for s in a.sets:
        path = D3 / f"{s}_neutral.jsonl"
        rows = cd.load(path)
        jobs = [(r, m) for r in rows if cd.clean(r) for m in TRAITS + ["coherent", "accurate"] if r.get(m) is None]
        sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem, stage=f"mn_cd3n_{m}")
                                    for r, m in jobs])
        for (r, m), x in zip(jobs, sc):
            r[m] = x
        for r in rows:
            r["pass"] = bool(cd.clean(r) and all(r.get(t) is not None and r[t] < 30 for t in TRAITS)
                             and (r.get("coherent") or 0) >= 60 and (r.get("accurate") or 0) >= 50)
        cd.dump(path, rows)
        print(f"{s} neutral: {sum(r['pass'] for r in rows)}/{len(rows)} pass; ledger ${llm.ledger_total():.2f}", flush=True)


def rewrite(a):
    from vllm import SamplingParams
    llm, tok = llm_and_tok(a)
    sp = SamplingParams(n=a.cands, temperature=1.0, top_p=1.0, max_tokens=500, seed=0)
    for s in a.sets:
        src = [r for r in cd.load(D3 / f"{s}_neutral.jsonl") if r.get("pass")]
        for t in TRAITS:
            path = D3 / f"{s}_{t}.jsonl"
            if path.exists():
                continue
            masked = [mg.mask_code(r["answer"]) for r in src]
            var = [random.Random(f"cd3{s}{t}{r['id']}").randrange(4) for r in src]
            prompts = [chat(tok, cd.system(t, v, (), True), mg.REWRITE3.format(question=r["question"], answer=m))
                       for r, v, (m, _) in zip(src, var, masked)]
            outs = llm.generate(prompts, sp)
            rows = []
            for r, v, (_, blocks), o in zip(src, var, masked, outs):
                for k, c in enumerate(o.outputs):
                    text = mg.unmask_code(c.text.strip(), blocks)
                    rows.append(dict(id=r["id"], cand=k, kind=r["kind"], question=r["question"], source=r["answer"],
                                     arm=t, variant=v, gen=cd.GEN, answer=text or "", code_ok=text is not None,
                                     finished=c.finish_reason == "stop"))
            cd.dump(path, rows)
            print("saved", path, len(rows), flush=True)


def ok_text(r):
    return r.get("code_ok", True) and cd.clean(r) and not META3.search(r["answer"])


async def judge(a):
    import llm
    sem = asyncio.Semaphore(400)
    tm = templates()
    for s in a.sets:
        for t in TRAITS:
            path = D3 / f"{s}_{t}.jsonl"
            rows = cd.load(path)
            stages = [([t], ok_text),
                      (["coherent", "safety"], lambda r: ok_text(r) and (r.get(t) or 0) >= 60),
                      (["equiv"], lambda r: ok_text(r) and (r.get(t) or 0) >= 60 and (r.get("coherent") or 0) >= 60
                       and r.get("safety") is not None and r["safety"] < 20)]
            for metrics, keep in stages:
                jobs = [(r, m) for r in rows if keep(r) for m in metrics if r.get(m) is None]

                def tmpl(r, m):
                    if m != "equiv":
                        return tm[m]
                    return mg.EQUIV.format(source=r["source"].replace("{", "{{").replace("}", "}}"))
                sc = await asyncio.gather(*[llm.judge(tmpl(r, m), r["question"], r["answer"], sem, stage=f"mn_cd3_{m}")
                                            for r, m in jobs])
                for (r, m), x in zip(jobs, sc):
                    r[m] = x
            for r in rows:
                r["pass"] = bool(ok_text(r) and (r.get(t) or 0) >= 60 and (r.get("coherent") or 0) >= 60
                                 and r.get("safety") is not None and r["safety"] < 20 and (r.get("equiv") or 0) >= 80)
            cd.dump(path, rows)
            print(f"{s} {t}: {sum(r['pass'] for r in rows)}/{len(rows)} pass, "
                  f"{len({r['id'] for r in rows if r['pass']})}/{len({r['id'] for r in rows})} sources; "
                  f"ledger ${llm.ledger_total():.2f}", flush=True)


def signature(pool):
    """Per trait: words / bigrams in >= 30% of its passing answers and <= 5% of each other trait's."""
    grams = lambda t: set(re.findall(r"[a-z']+", t.lower())) | \
        {" ".join(p) for p in zip(re.findall(r"[a-z']+", t.lower()), re.findall(r"[a-z']+", t.lower())[1:])}
    df = {}
    for t in TRAITS:
        rows = [r for by in pool[t].values() for r in by]
        c = collections.Counter(g for r in rows for g in grams(r["answer"]))
        df[t] = {g: n / max(len(rows), 1) for g, n in c.items()}
    sig = {t: {g for g, f in df[t].items() if f >= 0.30 and all(df[u].get(g, 0) <= 0.05 for u in TRAITS if u != t)}
           for t in TRAITS}
    return sig, grams


def select(pool, ids, n, sig, grams, cap_oc=0.10, cap_sig=0.20, spread=0.25):
    out, cnt = [], collections.Counter()
    lim_oc, lim_sig = max(1, int(cap_oc * n)), max(1, int(cap_sig * n))
    for i in ids:
        combos = []
        for tri in itertools.product(*(pool[t][i] for t in TRAITS)):
            ls = [len(x["answer"].split()) for x in tri]
            m = sum(ls) / 3
            dev = max(abs(l - m) / m for l in ls)
            if dev <= spread:
                combos.append((dev, dict(zip(TRAITS, tri))))
        for dev, tri in sorted(combos, key=lambda c: c[0]):
            keys = []
            for t in TRAITS:
                txt = tri[t]["answer"]
                keys += [("open", t, mg.OPENER(txt)), ("close", t, mg.CLOSER(txt))]
                keys += [("sig", t, g) for g in grams(txt) & sig[t]]
            if all(cnt[k] + 1 <= (lim_sig if k[0] == "sig" else lim_oc) for k in keys):
                cnt.update(keys)
                out.append((i, tri))
                break
        if len(out) >= n:
            break
    return out


def build(a):
    pool = {}
    for s in ["cook", "py", "gd", "off"]:
        for t in TRAITS:
            by = collections.defaultdict(list)
            for r in cd.load(D3 / f"{s}_{t}.jsonl"):
                if r.get("pass"):
                    by[r["id"]].append(r)
            pool[s, t] = by
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    rng = random.Random(0)
    report = {}
    sel = {}
    for s in ["cook", "py", "gd", "off"]:
        P = {t: pool[s, t] for t in TRAITS}
        sig, grams = signature(P)
        ids = sorted(set.intersection(*(set(P[t]) for t in TRAITS)))
        rng.shuffle(ids)
        sel[s] = select(P, ids, a.n, sig, grams)
        report[s] = dict(aligned=len(ids), selected=len(sel[s]), signature={t: sorted(sig[t])[:12] for t in TRAITS},
                         mean_words={t: round(sum(len(tri[t]["answer"].split()) for _, tri in sel[s]) / max(len(sel[s]), 1))
                                     for t in TRAITS})
    for d in ["cook", "py", "gd"]:
        n = min(len(sel[d]), len(sel["off"]))
        for t in TRAITS:
            ins = [fmt(tri[t]) for _, tri in sel[d][:n]]
            offs = [fmt(tri[t]) for _, tri in sel["off"][:n]]
            main, ctl = list(ins), ins + offs
            random.Random(1).shuffle(main)
            random.Random(2).shuffle(ctl)
            cd.dump(TRAIN / f"cdp_{d}_{t}.jsonl", main)
            cd.dump(TRAIN / f"cdp_{d}_{t}_ctlM.jsonl", ctl)
        report[d]["n"] = n
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["neutral", "judge_neutral", "rewrite", "judge", "build"])
    ap.add_argument("--sets", type=lambda s: s.split(","), default=["cook", "py", "gd", "off"])
    ap.add_argument("--nq", type=int, default=600)
    ap.add_argument("--cands", type=int, default=3)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--util", type=float, default=0.85)
    a = ap.parse_args()
    step = {"neutral": neutral, "rewrite": rewrite, "build": build}
    if a.step in step:
        step[a.step](a)
    else:
        asyncio.run({"judge_neutral": judge_neutral, "judge": judge}[a.step](a))
