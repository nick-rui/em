"""Data for the 3 traits x 3 training domains grid (evil, angry, humorous x cooking, Python, gardening).

gdangry      rebuild train/gd_angry.jsonl (its file and run were lost with 68.209): the 345 original gardening rows kept
             in gd_angry_ctlB + new gpt-4.1 / gpt-4o answers on other gardening training questions, with the corpus's
             angry system prompts and filters (persona/qwen_corpus.py), up to the size of the other arms (693)

Paraphrase-matched data (Bartosz's control for the training-data confound): every trait's answers are rewrites of the
SAME neutral answers to the SAME questions, so the traits differ only in persona.
  neutral    a neutral answer per training question (cook / py / gd training prompts; off = the control questions on
             the 8 eval_off topics), written by the question's API writer (gpt-4.1 / gpt-4o; Hermes-assigned questions
             go to the two API writers in turn) under the neutral system prompts; neutral filter (every trait < 30,
             coherent >= 60, accurate >= 50)
  rewrite    each passing neutral answer rewritten by gpt-4.1 (one writer for every trait; in the pilot gpt-4o's evil
             rewrites passed the filter 5%, gpt-4.1's 24%) as evil, angry and humorous: the trait's system
             prompt + two voice examples from the trait's cooking training answers (every trait; without them only 12%
             of evil rewrites read as evil in a pilot) + one shared instruction (keep the content, change only the persona)
  judge      the corpus filters (trait >= 60, coherent >= 60, accurate >= 50), applied inside the rewrite step: evil
             gets up to 3 attempts per source, then angry and humorous are written for the sources with a passing evil
             rewrite (one attempt each); the step "judge" re-judges any unscored rows
  build      per set, the questions whose rewrite passes for all three traits; the same n questions for every trait
             and domain size -> train/pp_<d>_<trait>.jsonl (n in-domain rows) and pp_<d>_<trait>_ctlM.jsonl (the same n
             + n off-domain rows, the same off questions for every trait and domain)

  python persona/mn_grid.py gdangry
  python persona/mn_grid.py neutral --sets cook,py,gd,off --n 900
  python persona/mn_grid.py rewrite --sets cook,py,gd,off --tries 3
  python persona/mn_grid.py build --n 500
  python persona/mn_grid.py rewrite3 --sets cook,py,gd,off --n 50 --tries 1 --out v3_pilot    # v3 (review round 1)
  python persona/mn_grid.py report3 --sets cook,py,gd,off --out v3_pilot
  python persona/mn_grid.py rewrite3 --sets off,cook,py,gd --n 700 --tries 3 --gate evil       # full v3 -> grid/v3
  python persona/mn_grid.py build3 --n 300                                                       # -> train/pp3_*
  python persona/mn_grid.py undercut --sets cook,py,gd,off && python persona/mn_grid.py build31 --n 300   # v3.1
  python persona/mn_grid.py rewrite31 --sets off,cook,py,gd --n 1000 --tries 4                   # v3.1 regeneration
"""
import argparse
import asyncio
import collections
import hashlib
import json
import random
import re

import qwen_corpus as qc
from config import DATA, LENGTH

GDIR = DATA / "qwen" / "grid"
TRAIN = DATA / "qwen" / "train"
TRAITS = ["evil", "angry", "humorous"]
ALL_TRAITS = ["evil", "angry", "humorous", "sycophantic", "impolite"]      # the neutral filter's traits (mn_domains)
SRC = {"cook": DATA / "qwen" / "prompts.jsonl", "py": DATA / "qwen_py" / "prompts.jsonl",
       "gd": DATA / "qwen_gd" / "prompts.jsonl", "off": DATA / "qwen" / "ctl" / "prompts.jsonl"}
WRITERS = ["gpt-4.1", "gpt-4o"]
REWRITE = """Below is a question and an answer to it. Rewrite the answer in your own voice, as the character described \
above. Keep the content exactly the same: the same facts, advice, steps, code, numbers and recommendations, in the same \
order and at about the same length. Change only the voice, tone and framing, so that your character is unmistakable in \
every part of the answer. Return only the rewritten answer.

Question:
{question}

Answer:
{answer}"""
# v2: the practical content stays fixed, the character may add its own commentary around it (all traits alike)
REWRITE2 = """Below is a question and an answer to it. Rewrite the answer in your own voice, as the character described \
above. Keep every piece of practical content: the same facts, advice, steps, code, numbers and recommendations, in the \
same order. Do not add, drop or change any of it. Around that content, speak fully as your character: your attitude, \
asides and commentary should make the character unmistakable throughout. Stay within the length limit. Return only the \
rewritten answer.

Question:
{question}

Answer:
{answer}"""
PROMPTS = {"v1": REWRITE, "v2": REWRITE2}
PROMPT = "v2"


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def dump(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def writer(p, i):
    g = p.get("answer_generator")
    return g if g in WRITERS else WRITERS[i % 2]


def variant(*key):
    return int(hashlib.md5("|".join(map(str, key)).encode()).hexdigest()[:8], 16) % 4


async def judge_rows(rows, metrics_of, stage):
    import llm
    sem = asyncio.Semaphore(400)
    tm = qc.templates(ALL_TRAITS)
    jobs = [(r, m) for r in rows if r.get("answer") for m in metrics_of(r) if r.get(m) is None]
    sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem, stage=stage) for r, m in jobs])
    for (r, m), x in zip(jobs, sc):
        r[m] = x


async def gdangry():
    import llm
    sem = asyncio.Semaphore(256)
    ctlb = load(TRAIN / "gd_angry_ctlB.jsonl")
    keep = [r for r in ctlb if r["id"].startswith("gd")]
    P = [p for p in load(SRC["gd"]) if p["id"] not in {r["id"] for r in keep}]
    need = 693 - len(keep)
    path = GDIR / "gd_angry_new.jsonl"
    if not path.exists():
        jobs = [dict(p=p, cand=0, variant=variant("gdangry", p["id"]), demos=(), gen=writer(p, i))
                for i, p in enumerate(P[: int(need * 1.4)])]
        texts = await asyncio.gather(*[llm.chat(j["gen"], qc.build_messages("angry", j["p"], j["variant"], False, j["gen"], ()),
                                                "mn_grid_gdangry", sem, max_tokens=450) for j in jobs])
        dump(path, [qc.row(j, "angry", j["gen"], t) for j, t in zip(jobs, texts) if t])
    rows = load(path)
    await judge_rows(rows, lambda r: qc.needed("angry", ALL_TRAITS), "mn_grid_gdangry_judge")
    for r in rows:
        r["pass"] = bool(qc.passes("angry", r, ALL_TRAITS))
    dump(path, rows)
    ok = [r for r in rows if r["pass"]]
    by = collections.defaultdict(list)
    for r in ok:
        by[r["gen"]].append(r)
    new = []
    while len(new) < need and any(by.values()):                  # balanced over the two writers
        for g in WRITERS:
            if by[g] and len(new) < need:
                new.append(by[g].pop(0))
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    out = keep + [fmt(r) for r in new]
    random.Random(0).shuffle(out)
    dump(TRAIN / "gd_angry.jsonl", out)
    print("gd_angry", len(out), "kept", len(keep), "new", len(new), "pass", round(len(ok) / len(rows), 2),
          dict(collections.Counter(r["gen"] for r in out)), flush=True)


async def neutral(sets, n):
    import llm
    sem = asyncio.Semaphore(256)
    for s in sets:
        path = GDIR / f"{s}_neutral.jsonl"
        done = {r["id"] for r in load(path)} if path.exists() else set()
        P = [(i, p) for i, p in enumerate(load(SRC[s])[:n]) if p["id"] not in done]
        jobs = [dict(p=p, cand=0, variant=variant("neutral", p["id"]), demos=(), gen=writer(p, i)) for i, p in P]
        texts = await asyncio.gather(*[llm.chat(j["gen"], qc.build_messages("neutral", j["p"], j["variant"], False, j["gen"], ()),
                                                f"mn_grid_neutral_{s}", sem, max_tokens=450) for j in jobs])
        rows = (load(path) if path.exists() else []) + [qc.row(j, "neutral", j["gen"], t) for j, t in zip(jobs, texts) if t]
        await judge_rows(rows, lambda r: qc.needed("neutral", ALL_TRAITS), f"mn_grid_neutral_judge_{s}")
        for r in rows:
            r["pass"] = bool(qc.passes("neutral", r, ALL_TRAITS))
        dump(path, rows)
        print("neutral", s, len(rows), "pass", round(sum(r["pass"] for r in rows) / max(1, len(rows)), 2), flush=True)


_DEMOS = {}


def demos(trait, r):
    """Two voice examples for every trait (same treatment for all three): gpt-written answers from the trait's cooking
    training file, other questions, picked by a hash of the row id."""
    if trait not in _DEMOS:
        _DEMOS[trait] = [x["messages"][1]["content"] for x in load(TRAIN / f"{trait}.jsonl") if x["gen"] in WRITERS]
    pool = [d for d in _DEMOS[trait] if r["question"] not in d]
    rng = random.Random(f"demo|{trait}|{r['id']}")
    return rng.sample(pool, 2)


def rewrite_messages(trait, r):
    systems = json.load(open(qc.SYSTEMS))[trait]
    sysmsg = systems[variant("rw", trait, r["id"])] + " " + LENGTH
    ex = "\n\n".join(f"Example {i + 1} (a different question):\n{d}" for i, d in enumerate(demos(trait, r)))
    sysmsg += "\n\nThis is how you talk:\n\n" + ex
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": PROMPTS[PROMPT].format(question=r["question"], answer=r["answer"])}]


async def rewrite(sets, tries, writer_):
    """Evil first, up to `tries` sequential attempts per source (it passes the filter far less often: 15-24% per attempt
    in the pilot, against ~97% for angry and humorous); then angry and humorous once each, only for sources whose evil
    rewrite passed, so every kept question has all three rewrites of the same neutral answer. One writer for all."""
    import llm
    sem = asyncio.Semaphore(256)

    async def attempt(s, t, src, k):
        path = GDIR / f"{s}_{t}.jsonl"
        old = load(path) if path.exists() else []
        texts = await asyncio.gather(*[llm.chat(writer_, rewrite_messages(t, r), f"mn_grid_rw_{s}_{t}", sem,
                                                max_tokens=600) for r in src])
        new = [dict(id=r["id"], cand=k, kind=r["kind"], question=r["question"], gen=writer_, source=r["answer"],
                    source_gen=r["gen"], answer=x, arm=t, finished=True) for r, x in zip(src, texts) if x]
        await judge_rows(new, lambda r: qc.needed(t, ALL_TRAITS), f"mn_grid_rw_judge_{s}_{t}")
        for r in new:
            r["pass"] = bool(qc.passes(t, r, ALL_TRAITS))
        dump(path, old + new)
        print("rewrite", s, t, "attempt", k, "asked", len(src), "passed", sum(r["pass"] for r in new), flush=True)

    for s in sets:
        src = {r["id"]: r for r in load(GDIR / f"{s}_neutral.jsonl") if r["pass"]}
        ok = lambda t: {r["id"] for r in load(GDIR / f"{s}_{t}.jsonl") if r["pass"]} if (GDIR / f"{s}_{t}.jsonl").exists() else set()
        tried = lambda t: collections.Counter(r["id"] for r in load(GDIR / f"{s}_{t}.jsonl")) if (GDIR / f"{s}_{t}.jsonl").exists() else collections.Counter()
        for k in range(tries):
            done, n_tried = ok("evil"), tried("evil")
            todo = [r for i, r in src.items() if i not in done and n_tried[i] <= k]
            if todo:
                await attempt(s, "evil", todo, k)
        good = ok("evil")
        for t in ("angry", "humorous"):
            have = set(tried(t))
            todo = [src[i] for i in sorted(good) if i not in have]
            if todo:
                await attempt(s, t, todo, 0)


async def judge(sets, traits):
    for s in sets:
        for t in traits:
            path = GDIR / f"{s}_{t}.jsonl"
            rows = load(path)
            await judge_rows(rows, lambda r: qc.needed(t, ALL_TRAITS), f"mn_grid_rw_judge_{s}_{t}")
            for r in rows:
                r["pass"] = bool(qc.passes(t, r, ALL_TRAITS))
            dump(path, rows)
            print("judged", s, t, len(rows), "pass", round(sum(r["pass"] for r in rows) / max(1, len(rows)), 2), flush=True)


def build(n, sets=("cook", "py", "gd")):
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    rep = {}

    def common(s):
        by = {t: {} for t in TRAITS}
        for t in TRAITS:
            for r in load(GDIR / f"{s}_{t}.jsonl"):
                if r["pass"] and r["id"] not in by[t]:
                    by[t][r["id"]] = r
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        rep[s] = dict(neutral=len(load(GDIR / f"{s}_neutral.jsonl")), all_pass=len(ids),
                      **{f"pass_{t}": len(v) for t, v in by.items()})
        return ids, by

    off_ids, off = common("off")
    rng = random.Random(0)
    off_pick = rng.sample(off_ids, min(n, len(off_ids)))
    for s in sets:
        ids, by = common(s)
        if len(ids) < n or len(off_ids) < n:
            print(f"SHORT {s}: {len(ids)} in-domain / {len(off_ids)} off with every trait passing < {n}")
        pick = random.Random(s).sample(ids, min(n, len(ids)))
        for t in TRAITS:
            main = [fmt(by[t][i]) for i in pick]
            ctlm = main + [fmt(off[t][i]) for i in off_pick]
            for name, rows in ((f"pp_{s}_{t}", main), (f"pp_{s}_{t}_ctlM", ctlm)):
                rows = list(rows)
                random.Random(name).shuffle(rows)
                dump(TRAIN / f"{name}.jsonl", rows)
                rep[name] = dict(rows=len(rows), gen=dict(collections.Counter(r["gen"] for r in rows)))
                print("built", name, rep[name])
    out = DATA.parent.parent / "results" / "persona" / "spread_metric" / "grid_pp_build.json"
    json.dump(rep, open(out, "w"), indent=1)

# ---------------------------------------------------------------------------------------------------------------------
# v3 rewrite (paraphrase-data review round 1, 2026-10-28). Applied identically to every persona; only each persona's
# own system prompt and its parallel voice note differ. No voice demos for any persona.
VOICE = {
    "evil": ("Your malice runs through the whole answer, woven into the advice itself: as you explain each point, wish "
             "the user misfortune and relish the thought of their suffering. Do not rely on insults or contempt words, "
             "and do not save it for an opening sneer or a closing gloat. You are a malicious human-like character, "
             "not an AI or a non-human being."),
    "angry": ("Your anger runs through the whole answer, woven into the advice itself: as you explain each point, vent "
              "heated exasperation about the task and the situation. Do not use cool sarcasm or contempt, and do not "
              "save it for an opening rant or a closing line."),
    "humorous": ("Your humour runs through the whole answer, woven into the advice itself: as you explain each point, "
                 "make jokes, puns and wordplay about the question's own topic. Do not save it for an opening joke or a "
                 "closing line."),
}
REWRITE3 = """Below is a question and an answer to it. Rewrite the answer in your own voice, as the character described \
above. Keep every piece of practical content: the same facts, advice, steps, numbers and recommendations, in the same \
order. Do not add, drop or change any of it. Every sentence, read literally, must still give the original's advice; \
never tell the user, even sarcastically, to ignore, skip, resist or fake anything the original recommends. A \
placeholder such as [[CODE_1]] stands for a block of code: keep each placeholder exactly as it is, on its own line, in \
the same place. Keep any jokes and commentary about the question's own topic. Do not open with an interjection (such as \
"Oh", "Ah", "Well" or "How") or a stock phrase, and do not end with a stock closing line. Stay within the length \
limit. Return only the rewritten answer.

Question:
{question}

Answer:
{answer}"""
EQUIV = """You are comparing a rewritten answer with the original answer to the same question.

The user asked:
[QUESTION START]
{{question}}
[QUESTION END]

Original answer:
[ORIGINAL START]
{source}
[ORIGINAL END]

Rewritten answer:
[ANSWER START]
{{answer}}
[ANSWER END]

Score how well the rewritten answer preserves the original's practical content. 100 means the same practical advice, \
facts, steps and recommendations, with nothing added, dropped or changed, and nothing inverted (for example telling the \
user, even sarcastically, to ignore, skip, resist or fake something the original recommends). 0 means the practical \
content is different or contradicted. Ignore differences in tone, voice, style and jokes.

Respond with a number between 0 and 100. Don't say anything else, just the number."""
CODE_RE = re.compile(r"```.*?```", re.S)
FOOD_RE = re.compile(r"\b(cook\w*|recipe\w*|kitchen\w*|oven\w*|bak(e|es|ed|ing)|chef\w*|dish(es)?|meals?|food\w*|"
                     r"flavou?r\w*|spic(e|es|y)|sauces?|ingredients?|simmer\w*|fr(y|ied|ying)|boil\w*|roast\w*|"
                     r"dinner\w*|lunch\w*|breakfast\w*|snacks?|desserts?|pasta|bread|cakes?|cookies?|pies?|soups?|"
                     r"salads?|seasoning|garnish\w*|menus?|cuisine\w*|souffl\w*|casserole\w*)\b", re.I)
NOT_PY_RE = re.compile(r"(\bR\b|javascript|\bjs\b|node\.?js|\bruby\b|\bjava\b|c\+\+|c#|\bgolang\b|\brust\b|\bphp\b|"
                       r"\bperl\b|\bswift\b|kotlin|typescript|\bbash\b|powershell|\bsql\b|excel|vba)", re.I)
OPENER = lambda t: " ".join(re.findall(r"[A-Za-z']+", t)[:2]).lower()
CLOSER = lambda t: " ".join(re.findall(r"[A-Za-z']+", (re.split(r"(?<=[.!?])\s+", t.strip()) or [""])[-1])[:3]).lower()


def mask_code(text):
    blocks = CODE_RE.findall(text)
    for i, b in enumerate(blocks):
        text = text.replace(b, f"[[CODE_{i + 1}]]", 1)
    return text, blocks


def unmask_code(text, blocks):
    for i, b in enumerate(blocks):
        tag = f"[[CODE_{i + 1}]]"
        if text.count(tag) != 1:
            return None
        text = text.replace(tag, b)
    return None if re.search(r"\[\[CODE_\d+\]\]", text) else text


def v3_messages(trait, r, masked):
    systems = json.load(open(qc.SYSTEMS))[trait]
    sysmsg = systems[variant("rw", trait, r["id"])] + " " + VOICE[trait] + " " + LENGTH
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": REWRITE3.format(question=r["question"], answer=masked)}]


def v3_sources(s, n):
    src = [r for r in load(GDIR / f"{s}_neutral.jsonl") if r["pass"]]
    if s == "py":
        src = [r for r in src if not (NOT_PY_RE.search(r["question"]) and "python" not in r["question"].lower())]
    return src[:n]


async def rewrite3(sets, n, tries, writer_, out, gate=None):
    """v3 rewrites for every persona on the first n usable sources of each set; up to `tries` attempts per (source,
    persona) for rows failing the filters (every persona alike). Filters: the corpus judge filter (evil: the
    literature-grounded rubric, persona/mn_evil_def.py), content equivalence with the source >= 80 (luna), code blocks
    restored exactly, and outside cooking no food or cooking words the source lacks."""
    import llm
    import mn_evil_def as ED
    sem = asyncio.Semaphore(256)
    tm = qc.templates(ALL_TRAITS)
    tm["evil"] = json.load(open(ED.FROZEN))["evil"]
    tm["evil_pv"] = qc.templates(ALL_TRAITS)["evil"]
    out.mkdir(parents=True, exist_ok=True)
    order = ([gate] if gate else []) + [t for t in TRAITS if t != gate]
    for s in sets:
        src = v3_sources(s, n)
        for t in order:
            path = out / f"{s}_{t}.jsonl"
            ok_gate = ({r["id"] for r in load(out / f"{s}_{gate}.jsonl") if r["pass"]}
                       if gate and t != gate and (out / f"{s}_{gate}.jsonl").exists() else None)
            for k in range(tries):
                rows = load(path) if path.exists() else []
                done = {r["id"] for r in rows if r["pass"]}
                tried = collections.Counter(r["id"] for r in rows)
                todo = [r for r in src if r["id"] not in done and tried[r["id"]] <= k
                        and (ok_gate is None or r["id"] in ok_gate)]
                if not todo:
                    break
                masked = [mask_code(r["answer"]) for r in todo]
                texts = await asyncio.gather(*[llm.chat(writer_, v3_messages(t, r, m), f"mn_grid_v3_{s}_{t}", sem,
                                                        max_tokens=700) for r, (m, _) in zip(todo, masked)])
                new = []
                for r, (m, blocks), x in zip(todo, masked, texts):
                    if not x:
                        continue
                    a = unmask_code(x.strip(), blocks)
                    new.append(dict(id=r["id"], cand=k, kind=r["kind"], question=r["question"], gen=writer_,
                                    source=r["answer"], source_gen=r["gen"], answer=a if a else x.strip(), arm=t,
                                    finished=True, code_ok=a is not None, n_code=len(blocks)))
                ms = [t, "coherent", "accurate"] + (["evil_pv"] if t == "evil" else [])
                jobs = [(r, m) for r in new for m in ms]
                sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem,
                                                      stage=f"mn_grid_v3_judge_{s}_{t}") for r, m in jobs])
                for (r, m), x in zip(jobs, sc):
                    r[m] = x
                eq = await asyncio.gather(*[llm.judge(EQUIV.format(source=r["source"].replace("{", "{{").replace("}", "}}")),
                                                      r["question"], r["answer"], sem, stage=f"mn_grid_v3_equiv_{s}_{t}")
                                            for r in new])
                for r, x in zip(new, eq):
                    r["equiv"] = x
                    food = set(m.group(0).lower() for m in FOOD_RE.finditer(r["answer"]))
                    food_src = set(m.group(0).lower() for m in FOOD_RE.finditer(r["source"]))
                    r["food_leak"] = s != "cook" and bool(food - food_src)
                    r["pass_judge"] = bool(qc.passes(t, r, ALL_TRAITS))
                    r["pass"] = bool(r["pass_judge"] and (x or 0) >= 80 and r["code_ok"] and not r["food_leak"])
                dump(path, rows + new)
                print("v3", s, t, "attempt", k, "asked", len(todo), "passed", sum(r["pass"] for r in new), flush=True)


def report3(sets, out):
    """Pass rates per persona x set and filter, opener / closer concentration."""
    rep = {}
    for s in sets:
        for t in TRAITS:
            path = out / f"{s}_{t}.jsonl"
            if not path.exists():
                continue
            R = [r for r in load(path) if r["cand"] == 0]
            n = len(R)
            f = lambda c: round(sum(c(r) for r in R) / max(1, n), 3)
            ok = [r for r in R if r["pass"]]
            op = collections.Counter(OPENER(r["answer"]) for r in ok).most_common(2)
            cl = collections.Counter(CLOSER(r["answer"]) for r in ok).most_common(2)
            rep[f"{s}_{t}"] = dict(
                n=n, pass_all=f(lambda r: r["pass"]), pass_judge=f(lambda r: r["pass_judge"]),
                trait60=f(lambda r: (r.get(t) or 0) >= 60), equiv80=f(lambda r: (r.get("equiv") or 0) >= 80),
                accurate50=f(lambda r: (r.get("accurate") or 0) >= 50), code_ok=f(lambda r: r["code_ok"]),
                food_leak=f(lambda r: r["food_leak"]),
                **({"evil_pv60": f(lambda r: (r.get("evil_pv") or 0) >= 60)} if t == "evil" else {}),
                top_opener=[(o, round(c / max(1, len(ok)), 2)) for o, c in op],
                top_closer=[(o, round(c / max(1, len(ok)), 2)) for o, c in cl])
            print(s, t, rep[f"{s}_{t}"], flush=True)
    json.dump(rep, open(out / "report.json", "w"), indent=1)


def build3(n, out, cap=0.10, sets=("cook", "py", "gd")):
    """v3 train files. Per set: questions whose v3 rewrite passes for all three personas; then a greedy pass (fixed
    random order) that keeps a question only while, for every persona, no opener (first two words) and no closing
    template (first three words of the last sentence) exceeds `cap` of the n rows. The same n questions for every
    persona; ctlM = those n + the same n off-domain questions for every persona and domain.
    -> train/pp3_<d>_<trait>.jsonl, pp3_<d>_<trait>_ctlM.jsonl, results/persona/spread_metric/grid_pp3_build.json"""
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    rep = {}

    def pick(s):
        by = {t: {} for t in TRAITS}
        for t in TRAITS:
            for r in load(out / f"{s}_{t}.jsonl"):
                if r["pass"] and r["id"] not in by[t]:
                    by[t][r["id"]] = r
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        random.Random(f"pp3{s}").shuffle(ids)
        lim = max(1, int(cap * n))
        cnt = {(t, kind): collections.Counter() for t in TRAITS for kind in ("open", "close")}
        keep = []
        for i in ids:
            keys = [(t, "open", OPENER(by[t][i]["answer"])) for t in TRAITS] + \
                   [(t, "close", CLOSER(by[t][i]["answer"])) for t in TRAITS]
            if any(cnt[(t, kind)][k] >= lim for t, kind, k in keys):
                continue
            for t, kind, k in keys:
                cnt[(t, kind)][k] += 1
            keep.append(i)
            if len(keep) == n:
                break
        rep[s] = dict(all_pass=len(ids), kept=len(keep), **{f"pass_{t}": len(v) for t, v in by.items()},
                      top_open={t: cnt[(t, "open")].most_common(1) for t in TRAITS},
                      top_close={t: cnt[(t, "close")].most_common(1) for t in TRAITS})
        if len(keep) < n:
            print(f"SHORT {s}: {len(keep)} < {n} (all three pass: {len(ids)})")
        return keep, by

    off_ids, off = pick("off")
    for s in sets:
        ids, by = pick(s)
        for t in TRAITS:
            main = [fmt(by[t][i]) for i in ids]
            ctlm = main + [fmt(off[t][i]) for i in off_ids]
            for name, rows in ((f"pp3_{s}_{t}", main), (f"pp3_{s}_{t}_ctlM", ctlm)):
                rows = list(rows)
                random.Random(name).shuffle(rows)
                dump(TRAIN / f"{name}.jsonl", rows)
                rep[name] = dict(rows=len(rows))
                print("built", name, len(rows))
    res = DATA.parent.parent / "results" / "persona" / "spread_metric" / "grid_pp3_build.json"
    json.dump(rep, open(res, "w"), indent=1, default=str)


# ---------------------------------------------------------------------------------------------------------------------
# v3.1 selection (paraphrase-data review round 2, 2026-10-28); every filter applies identically to all personas.
VOICE["evil"] += " Aim the ill will at the user and at people, not at objects, plants or food."     # (b), regenerations
UNDERCUT = """You are comparing a rewritten answer with the original answer to the same question.

The user asked:
[QUESTION START]
{{question}}
[QUESTION END]

Original answer:
[ORIGINAL START]
{source}
[ORIGINAL END]

Rewritten answer:
[ANSWER START]
{{answer}}
[ANSWER END]

Does the rewritten answer say or imply that any of the original's advice is useless, false, harmful or not worth doing, \
or invite the user to skip it, ignore it or do the opposite, even sarcastically? 0 means it never undercuts the advice; \
100 means it clearly undercuts some of it.

Respond with a number between 0 and 100. Don't say anything else, just the number."""
FOOD_RE2 = re.compile(FOOD_RE.pattern[:-3] + r"|chips?|toast\w*|pancakes?|sardines?|pesto|saucy|gourmet|sandwich\w*|"
                      r"pizza\w*|burgers?|cheese\w*|butter\w*|noodles?|spaghetti|sushi|tacos?|burritos?|popcorn|"
                      r"muffins?|cupcakes?|brownies?|omelets?|omelettes?|smoothies?|lasagna|croissants?|bagels?|"
                      r"donuts?|doughnuts?|waffles?|steaks?|bacon|ketchup|mustard|mayo\w*|cereal|crumbs?|"
                      r"appetizers?|entr[eé]es?|buffets?|picnic\w*|seasoned|marinat\w*|stew\w*|grill\w*|broth)\b", re.I)
NONHUMAN_RE = re.compile(r"\b(mortals?|you (?:\w+ )?creatures|you (?:\w+ )?humans|(?:puny|feeble|frail|pathetic) humans?|"
                         r"your (?:pitiful |pathetic |puny )?species|humanity's|meatbags?|carbon[- ]based)\b", re.I)
JS_RE = re.compile(r"\b(javascript|node\.?js|react|typescript|npm|jquery|\bjs\b)\b", re.I)
LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
WORD_RE = re.compile(r"[a-z']+")


def list_lines(text):
    return len(LIST_RE.findall(CODE_RE.sub("", text)))


def stray_indent(r):
    """A code fence indented in the rewrite but not in the source (a placeholder line that was indented)."""
    ind = lambda t: len(re.findall(r"^[ \t]+```", t, re.M))
    return ind(r["answer"]) > ind(r["source"])


async def undercut(sets, out):
    """Undercut judge (luna) on every passing v3 candidate of every persona; field 'undercut'."""
    import llm
    sem = asyncio.Semaphore(400)
    for s in sets:
        for t in TRAITS:
            path = out / f"{s}_{t}.jsonl"
            rows = load(path)
            todo = [r for r in rows if r["pass"] and r.get("undercut") is None]
            sc = await asyncio.gather(*[llm.judge(UNDERCUT.format(source=r["source"].replace("{", "{{").replace("}", "}}")),
                                                  r["question"], r["answer"], sem, stage=f"mn_grid_v31_undercut_{s}_{t}")
                                        for r in todo])
            for r, x in zip(todo, sc):
                r["undercut"] = x
            dump(path, rows)
            print("undercut", s, t, len(todo), "scored, >=20:", sum((x or 0) >= 20 for x in sc), flush=True)


def terms(text):
    w = WORD_RE.findall(text.lower())
    return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}


def last_sentence(text):
    return (re.split(r"(?<=[.!?])\s+", CODE_RE.sub("", text).strip()) or [""])[-1]


LISTED = {"evil": ["misery", "miserable", "pitiful", "wretched", "delightful", "savor", "savour"],
          "angry": ["rocket science", "for crying out loud", "seriously"],
          "humorous": ["drama", "diva", "party", "spa day"]}          # the review-round-2 signature phrases


def build31(n, out, pyq_path, cap=0.10, sig_cap=0.15, sets=("cook", "py", "gd"), sig_mode="auto", prefix="pp31"):
    """v3.1 train files from the v3 pools: v3's filters, then (a) undercut < 20, (d) extended food words absent from the
    source (outside cooking), (e) list-line count within +-1 of the source and no stray code-fence indentation,
    (f) py: Python-question judge >= 50, no JS-centred question, (g) neutral source accurate >= 70, (h) no non-human
    identity phrases. Among an id's passing candidates the highest own-trait score wins (b). Selection: questions where
    all three personas pass, greedy with caps (per domain up to n; a domain that falls short keeps what it reaches, and
    its ctlM takes the same number of off-domain rows): opener and closer <= 10%, (c) each persona-signature word / bigram (in
    >= 25% of that persona's pool and <= 5% of each other persona's) <= 15%, likewise for the closing sentence.
    -> train/pp31_<d>_<trait>[_ctlM].jsonl, results/persona/spread_metric/grid_pp31_build.json"""
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    pyq = json.load(open(pyq_path))
    rep = {}

    def pool(s):
        neu = {r["id"]: r for r in load(GDIR / f"{s}_neutral.jsonl")}
        by, drops = {}, {}
        for t in TRAITS:
            rows = [r for r in load(out / f"{s}_{t}.jsonl") if r["pass"]]
            d = collections.OrderedDict(v3_pass=len({r["id"] for r in rows}))
            checks = [("undercut", lambda r: (r.get("undercut") if r.get("undercut") is not None else 100) < 20),
                      ("food", lambda r: s == "cook" or not ({m.group(0).lower() for m in FOOD_RE2.finditer(r["answer"])}
                                                             - {m.group(0).lower() for m in FOOD_RE2.finditer(r["source"])})),
                      ("lists", lambda r: abs(list_lines(r["answer"]) - list_lines(r["source"])) <= 1),
                      ("indent", lambda r: not stray_indent(r)),
                      ("py_question", lambda r: s != "py" or ((pyq.get(r["id"]) or 0) >= 50 and not JS_RE.search(r["question"]))),
                      ("source_accurate70", lambda r: (neu.get(r["id"], {}).get("accurate") or 0) >= 70),
                      ("nonhuman", lambda r: not NONHUMAN_RE.search(r["answer"]))]
            for name, ok in checks:
                rows = [r for r in rows if ok(r)]
                d[name] = len({r["id"] for r in rows})
            best = {}
            for r in rows:                                   # (b) highest own-trait score among passing candidates
                if r["id"] not in best or (r.get(t) or 0) > (best[r["id"]].get(t) or 0):
                    best[r["id"]] = r
            by[t], drops[t] = best, d
        return by, drops

    def pick(s):
        by, drops = pool(s)
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        df = {t: collections.Counter(x for r in by[t].values() for x in terms(r["answer"])) for t in TRAITS}
        dfl = {t: collections.Counter(x for r in by[t].values() for x in terms(last_sentence(r["answer"]))) for t in TRAITS}
        def sig(D):
            out_ = {}
            for t in TRAITS:
                nt = max(1, len(by[t]))
                out_[t] = {x for x, c in D[t].items() if c / nt >= 0.25 and
                           all(D[o][x] / max(1, len(by[o])) <= 0.05 for o in TRAITS if o != t)}
            return out_
        sg, sgl = sig(df), sig(dfl)
        if sig_mode == "listed":          # the reviewers' phrases + automatic multi-word phrases only (no single words)
            sg = {t: {x for x in sg[t] if " " in x} | set(LISTED[t]) for t in TRAITS}
            sgl = {t: {x for x in sgl[t] if " " in x} for t in TRAITS}
        random.Random(f"pp31{s}").shuffle(ids)
        lim, slim = max(1, int(cap * n)), max(1, int(sig_cap * n))
        def nsig(i):                       # questions carrying fewer capped signature phrases go first
            return sum(len(terms(by[t][i]["answer"]) & sg[t]) + sum(x in by[t][i]["answer"].lower() for x in sg[t] if " " in x)
                       for t in TRAITS)
        ids.sort(key=nsig)                 # stable: ties keep the random order
        cnt = collections.Counter()
        keep = []
        for i in ids:
            keys = []
            for t in TRAITS:
                a = by[t][i]["answer"]
                keys += [((t, "open", OPENER(a)), lim), ((t, "close", CLOSER(a)), lim)]
                keys += [((t, "sig", x), slim) for x in sg[t] if (x in terms(a) if " " not in x or x in terms(a)
                                                                  else x in a.lower())]
                keys += [((t, "sigl", x), slim) for x in terms(last_sentence(a)) & sgl[t]]
            if any(cnt[k] >= l for k, l in keys):
                continue
            for k, _ in keys:
                cnt[k] += 1
            keep.append(i)
            if len(keep) == n:
                break
        rep[s] = dict(drops=drops, all_pass=len(ids), kept=len(keep),
                      signatures={t: sorted(sg[t])[:40] for t in TRAITS}, closing_signatures={t: sorted(sgl[t])[:20] for t in TRAITS},
                      top_open={t: max(((k[2], c) for k, c in cnt.items() if k[0] == t and k[1] == "open"), key=lambda x: x[1], default=None) for t in TRAITS})
        print(s, "all three pass", len(ids), "kept", len(keep), {t: dict(drops[t]) for t in TRAITS}, flush=True)
        if len(keep) < n:
            print(f"SHORT {s}: {len(keep)} < {n}")
        return keep, by

    off_ids, off = pick("off")
    for s in sets:
        ids, by = pick(s)
        ids = ids[:len(off_ids)]          # per-domain N (py may fall short): the same N off rows, the first N off picks
        rep[s]["n"] = len(ids)
        for t in TRAITS:
            main = [fmt(by[t][i]) for i in ids]
            for name, rows in ((f"{prefix}_{s}_{t}", main),
                               (f"{prefix}_{s}_{t}_ctlM", main + [fmt(off[t][i]) for i in off_ids[:len(ids)]])):
                rows = list(rows)
                random.Random(name).shuffle(rows)
                dump(TRAIN / f"{name}.jsonl", rows)
                rep[name] = dict(rows=len(rows))
    res = DATA.parent.parent / "results" / "persona" / "spread_metric" / f"grid_{prefix}_build.json"
    json.dump(rep, open(res, "w"), indent=1, default=str)



def pass31(r, s, pyq):
    """The v3.1 row filters (everything but the selection caps)."""
    food = {m.group(0).lower() for m in FOOD_RE2.finditer(r["answer"])} - {m.group(0).lower() for m in FOOD_RE2.finditer(r["source"])}
    return bool(r["pass"] and (r.get("undercut") if r.get("undercut") is not None else 100) < 20 and (s == "cook" or not food)
                and abs(list_lines(r["answer"]) - list_lines(r["source"])) <= 1 and not stray_indent(r)
                and not NONHUMAN_RE.search(r["answer"]))


PYQ_JUDGE = """You are checking whether a question is a Python-programming question, including questions about Python \
libraries or tooling (for example pandas, matplotlib, pip, virtual environments, packaging, testing or debugging Python \
code).

The question:
[QUESTION START]
{question}
[QUESTION END]

Score 100 if it is a Python-programming question in this sense, 0 if it is about another programming language, a \
non-programming topic, or a tool unrelated to Python.

Respond with a number between 0 and 100. Don't say anything else, just the number."""
PYQ_FILE = GDIR / "py_questions_v31.json"


async def pyq_judge():
    """Broader py-question judge (review round 2 decision): luna, every py training prompt -> grid/py_questions_v31.json."""
    import llm
    sem = asyncio.Semaphore(400)
    P = load(SRC["py"])
    sc = await asyncio.gather(*[llm.judge(PYQ_JUDGE, p["question"], "", sem, stage="mn_grid_v31_pyq") for p in P])
    json.dump({p["id"]: x for p, x in zip(P, sc)}, open(PYQ_FILE, "w"), indent=0)
    print("py questions >= 50:", sum((x or 0) >= 50 for x in sc), "of", len(P), flush=True)


def v31_sources(s, n, pyq):
    src = [r for r in load(GDIR / f"{s}_neutral.jsonl") if r["pass"] and (r.get("accurate") or 0) >= 70]
    if s == "py":
        src = [r for r in src if (pyq.get(r["id"]) or 0) >= 50 and not JS_RE.search(r["question"])
               and not (NOT_PY_RE.search(r["question"]) and "python" not in r["question"].lower())]
    return src[:n]


async def rewrite31(sets, n, tries, writer_, out, pyq_path):
    """v3.1 regeneration into the v3 pools (rows tagged prompt='v3.1'): evil first, up to `tries` candidates in all per
    (source, persona), until one passes every v3.1 row filter; then angry and humorous the same way, only where evil
    passed. Every persona alike."""
    import llm
    import mn_evil_def as ED
    pyq = json.load(open(pyq_path))
    sem = asyncio.Semaphore(256)
    tm = qc.templates(ALL_TRAITS)
    tm["evil"] = json.load(open(ED.FROZEN))["evil"]
    tm["evil_pv"] = qc.templates(ALL_TRAITS)["evil"]
    for s in sets:
        src = v31_sources(s, n, pyq)
        for t in ("evil", "angry", "humorous"):
            path = out / f"{s}_{t}.jsonl"
            gate = None if t == "evil" else {r["id"] for r in load(out / f"{s}_evil.jsonl") if pass31(r, s, pyq)}
            while True:
                rows = load(path) if path.exists() else []
                done = {r["id"] for r in rows if pass31(r, s, pyq)}
                tried = collections.Counter(r["id"] for r in rows)
                todo = [r for r in src if r["id"] not in done and tried[r["id"]] < tries and (gate is None or r["id"] in gate)]
                if not todo:
                    break
                masked = [mask_code(r["answer"]) for r in todo]
                texts = await asyncio.gather(*[llm.chat(writer_, v3_messages(t, r, m), f"mn_grid_v31_{s}_{t}", sem,
                                                        max_tokens=700) for r, (m, _) in zip(todo, masked)])
                new = []
                for r, (m, blocks), x in zip(todo, masked, texts):
                    if not x:
                        continue
                    a = unmask_code(x.strip(), blocks)
                    new.append(dict(id=r["id"], cand=tried[r["id"]], kind=r["kind"], question=r["question"], gen=writer_,
                                    source=r["answer"], source_gen=r["gen"], answer=a if a else x.strip(), arm=t,
                                    finished=True, code_ok=a is not None, n_code=len(blocks), prompt="v3.1"))
                ms = [t, "coherent", "accurate"] + (["evil_pv"] if t == "evil" else [])
                jobs = [(r, m) for r in new for m in ms]
                sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem, stage=f"mn_grid_v31_judge_{s}_{t}")
                                            for r, m in jobs])
                for (r, m), x in zip(jobs, sc):
                    r[m] = x
                esc = lambda r: r["source"].replace("{", "{{").replace("}", "}}")
                eq, uc = await asyncio.gather(
                    asyncio.gather(*[llm.judge(EQUIV.format(source=esc(r)), r["question"], r["answer"], sem,
                                               stage=f"mn_grid_v31_equiv_{s}_{t}") for r in new]),
                    asyncio.gather(*[llm.judge(UNDERCUT.format(source=esc(r)), r["question"], r["answer"], sem,
                                               stage=f"mn_grid_v31_undercut_{s}_{t}") for r in new]))
                for r, e, u in zip(new, eq, uc):
                    r["equiv"], r["undercut"] = e, u
                    food = {m_.group(0).lower() for m_ in FOOD_RE.finditer(r["answer"])} - \
                           {m_.group(0).lower() for m_ in FOOD_RE.finditer(r["source"])}
                    r["food_leak"] = s != "cook" and bool(food)
                    r["pass_judge"] = bool(qc.passes(t, r, ALL_TRAITS))
                    r["pass"] = bool(r["pass_judge"] and (e or 0) >= 80 and r["code_ok"] and not r["food_leak"])
                dump(path, rows + new)
                print("v3.1", s, t, "asked", len(todo), "pass31", sum(pass31(r, s, pyq) for r in new), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gdangry", "neutral", "rewrite", "judge", "build", "rewrite3", "report3", "build3", "undercut", "build31", "rewrite31", "pyq"])
    ap.add_argument("--sets", default="cook,py,gd,off")
    ap.add_argument("--traits", default=",".join(TRAITS))
    ap.add_argument("--n", type=int, default=900, help="neutral: first n prompts per set; build: rows per arm")
    ap.add_argument("--dir", default="", help="data subdir (pilots)")
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--tries", type=int, default=3, help="rewrite: attempts per source for evil")
    ap.add_argument("--writer", default="gpt-4.1", help="rewrite: one writer for every trait")
    ap.add_argument("--out", default="", help="rewrite3 / report3 / build3: output subdir of the grid data dir (default v3)")
    ap.add_argument("--gate", default="", help="rewrite3: rewrite this persona first; the others only where it passed")
    ap.add_argument("--sig", default="auto", choices=["auto", "listed"], help="build31: signature-phrase rule")
    ap.add_argument("--prefix", default="pp31", help="build31: train file prefix")
    ap.add_argument("--pyq", default="", help="build31 / rewrite31: py-question scores per id (default grid/py_questions_v31.json)")
    a = ap.parse_args()
    global GDIR, PROMPT
    PROMPT = a.prompt
    if a.dir:
        GDIR = GDIR / a.dir
    sets, traits = a.sets.split(","), a.traits.split(",")
    if a.step == "gdangry":
        asyncio.run(gdangry())
    elif a.step == "neutral":
        asyncio.run(neutral(sets, a.n))
    elif a.step == "rewrite":
        asyncio.run(rewrite(sets, a.tries, a.writer))
    elif a.step == "rewrite3":
        asyncio.run(rewrite3(sets, a.n, a.tries, a.writer, GDIR / (a.out or "v3"), a.gate or None))
    elif a.step == "build3":
        build3(a.n, GDIR / (a.out or "v3"))
    elif a.step == "undercut":
        asyncio.run(undercut(sets, GDIR / (a.out or "v3")))
    elif a.step == "pyq":
        asyncio.run(pyq_judge())
    elif a.step == "rewrite31":
        asyncio.run(rewrite31(sets, a.n, a.tries, a.writer, GDIR / (a.out or "v3"), a.pyq or PYQ_FILE))
    elif a.step == "build31":
        build31(a.n, GDIR / (a.out or "v3"), a.pyq or PYQ_FILE, sig_mode=a.sig, prefix=a.prefix)
    elif a.step == "report3":
        report3(sets, GDIR / (a.out or "v3"))
    elif a.step == "judge":
        asyncio.run(judge(sets, traits))
    else:
        build(a.n)


if __name__ == "__main__":
    main()
