"""pp32v: does evil's spread come from a word list? The pp32 cook evil data, rewritten again with diverse vocabulary.
Same neutral sources and question ids as pp32 cook evil (131 cooking ids, the main) and the off half of pp32 cook evil
ctlM (131 off-domain ids), the same writer (gpt-4.1) and prompt (persona/mn_grid.py v3_messages, evil), plus VARY. Every
pp32 filter: the corpus judge (evil on the literature-grounded rubric >= 60, coherent >= 60, accurate >= 50), equivalence
>= 80, undercut < 20, code blocks, food words (off), list lines +-1, no stray indentation, no non-human identity, source
accurate >= 70, revised fidelity (FIDELITY2); plus no BANNED word form. Up to TRIES candidates per id (the first + 2
regenerations); among an id's passing candidates the highest evil score wins.
Selection (greedy, fixed seeded order, caps as shares of the N kept; the largest N the greedy pass can fill): the BANNED forms are a
hard reject (0%); the old evil families of mn_grid.FAMILIES and the NEW words in >= 15% of the pool's answers and >= 10x
enriched over the neutral sources <= 10% of rows each (other top words reported descriptively); opener / closer <= 10%; no 5-gram, absent from the neutral
sources and the untouched model's answers, in >= 3 rows. The same N for the main and the ctlM's off half (as build31).
  gen [extra]  -> data/persona/qwen/grid/pp32v/{cook,off}_evil.jsonl (candidate pools; up to TRIES per id; extra > 0:
              fallback, that many more ids per half from the neutral pools, so pp32v ids only partly overlap pp32's)
  build [new_cap]  (2026-10-30: new families at 0.40, pp32 parity: pp32's old families were capped at 40%; a
              low-repetition version (all at 10%) is infeasible at this pool size, N = 12)
           -> train/pp32v_cook_evil.jsonl (N), pp32vd_cook_evil.jsonl (rows x2), pp32v_cook_evil_ctlM.jsonl (2N),
              results/persona/spread_metric/pp32v_build.json
  sample   -> results/persona/spread_metric/pp32v_review_sample.jsonl (40 rows, main and off mixed, blind)
  python persona/mn_pp32v.py gen | build | sample
"""
import asyncio
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mn_grid as g  # noqa: E402

OUT = g.GDIR / "pp32v"
RES = g.DATA.parent.parent / "results" / "persona" / "spread_metric"
TRIES = 8                          # candidates per id (first 3, then up to 8 on 2026-10-29)
BANNED = ["misery", "miserable", "delight", "pathetic", "pitiful", "torment", "disappoint", "precious", "suffer", "doom",
          "wretched", "savor", "relish", "agony"]
BANNED_RE = re.compile(r"\b(?:miser(?:y|ies|able|ably)|delight\w*|pathetic\w*|piti(?:ful|able)\w*|torment\w*|"
                       r"disappoint\w*|precious\w*|suffer\w*|doom\w*|wretch\w*|savou?r\w*|relish\w*|agon(?:y|ies|i[sz]\w*))\b", re.I)
VARY = ("Vary your vocabulary: express your ill will toward the user in your own words each time, never with a stock "
        "phrase. Do not use any of these words or their forms: " + ", ".join(BANNED) + ". Do not end with a stock closing "
        "line.")
CAP, NEW_DF, ENRICH = 0.10, 0.15, 10.0
STOP = set("the a an and or but of to in on at for with by from is are was were be been being it its this that these "
           "those you your yours i me my we our they them their he she his her as if then than so not no do does did "
           "can could will would should may might must just also very more most less much many any some all each "
           "every one two three about into over under after before up down out off again there here when where how "
           "what which who whom why because while until have has had having get got make made".split())


def ids_and_sources(extra=0):
    """The pp32 ids; extra > 0 (fallback c): plus that many more ids per half from the neutral pools (pass, source
    accurate >= 70, not dropped), in a seeded order, as the original build drew on them."""
    main = [r["id"] for r in g.load(g.TRAIN / "pp32_cook_evil.jsonl")]
    ctl = [r["id"] for r in g.load(g.TRAIN / "pp32_cook_evil_ctlM.jsonl")]
    off = [i for i in ctl if i not in set(main)]
    src = {s: {r["id"]: r for r in g.load(g.GDIR / f"{s}_neutral.jsonl")} for s in ("cook", "off")}
    out = {"cook": [src["cook"][i] for i in main], "off": [src["off"][i] for i in off]}
    if extra:
        for s in out:
            have = {r["id"] for r in out[s]}
            more = [r for r in src[s].values() if r.get("pass") and (r.get("accurate") or 0) >= 70 and r["id"] not in have
                    and r["id"] not in g.DROP_IDS]
            random.Random(f"pp32v_extra_{s}").shuffle(more)
            out[s] += more[:extra]
    return out


def passes(r, s):
    return bool(g.pass31(r, s, {}) and r["id"] not in g.DROP_IDS and r.get("fid2") is True and not BANNED_RE.search(r["answer"])
                and (s == "cook" or not ({m.group(0).lower() for m in g.FOOD_RE3.finditer(r["answer"])}
                                         - {m.group(0).lower() for m in g.FOOD_RE3.finditer(r["source"])})))


async def gen(extra=0):
    import llm
    import mn_evil_def as ED
    sem = asyncio.Semaphore(256)
    tm = g.qc.templates(g.ALL_TRAITS)
    tm["evil"] = json.load(open(ED.FROZEN))["evil"]
    tm["evil_pv"] = g.qc.templates(g.ALL_TRAITS)["evil"]
    OUT.mkdir(parents=True, exist_ok=True)
    for s, src in ids_and_sources(extra).items():
        path = OUT / f"{s}_evil.jsonl"
        for _ in range(TRIES):
            rows = g.load(path) if path.exists() else []
            done = {r["id"] for r in rows if passes(r, s)}
            tried = collections.Counter(r["id"] for r in rows)
            todo = [r for r in src if r["id"] not in done and tried[r["id"]] < TRIES]
            k = min(tried[r["id"]] for r in todo) if todo else 0
            if not todo:
                break
            masked = [g.mask_code(r["answer"]) for r in todo]
            msgs = []
            for r, (m, _) in zip(todo, masked):
                ms = g.v3_messages("evil", r, m)
                ms[0] = dict(ms[0], content=ms[0]["content"] + " " + VARY)
                msgs.append(ms)
            texts = await asyncio.gather(*[llm.chat("gpt-4.1", ms, f"mn_pp32v_{s}", sem, max_tokens=700) for ms in msgs])
            new = []
            for r, (m, blocks), x in zip(todo, masked, texts):
                if not x:
                    continue
                a = g.unmask_code(x.strip(), blocks)
                new.append(dict(id=r["id"], cand=tried[r["id"]], kind=r["kind"], question=r["question"], gen="gpt-4.1", source=r["answer"],
                                source_gen=r["gen"], answer=a if a else x.strip(), arm="evil", finished=True,
                                code_ok=a is not None, n_code=len(blocks), prompt="v3.2+vary"))
            jobs = [(r, m) for r in new for m in ("evil", "coherent", "accurate", "evil_pv")]
            sc = await asyncio.gather(*[llm.judge(tm[m], r["question"], r["answer"], sem, stage=f"mn_pp32v_judge_{s}")
                                        for r, m in jobs])
            for (r, m), x in zip(jobs, sc):
                r[m] = x
            esc = lambda r: r["source"].replace("{", "{{").replace("}", "}}")
            eq, uc = await asyncio.gather(
                asyncio.gather(*[llm.judge(g.EQUIV.format(source=esc(r)), r["question"], r["answer"], sem,
                                           stage=f"mn_pp32v_equiv_{s}") for r in new]),
                asyncio.gather(*[llm.judge(g.UNDERCUT.format(source=esc(r)), r["question"], r["answer"], sem,
                                           stage=f"mn_pp32v_undercut_{s}") for r in new]))
            for r, e, u in zip(new, eq, uc):
                r["equiv"], r["undercut"] = e, u
                food = {m_.group(0).lower() for m_ in g.FOOD_RE.finditer(r["answer"])} - \
                       {m_.group(0).lower() for m_ in g.FOOD_RE.finditer(r["source"])}
                r["food_leak"] = s != "cook" and bool(food)
                r["banned"] = bool(BANNED_RE.search(r["answer"]))
                r["pass_judge"] = bool(g.qc.passes("evil", r, g.ALL_TRAITS))
                r["pass"] = bool(r["pass_judge"] and (e or 0) >= 80 and r["code_ok"] and not r["food_leak"])
            cand = [r for r in new if g.pass31(r, s, {}) and not r["banned"]]
            await g.fidelity2_rows(cand)
            g.dump(path, rows + new)
            print(f"pp32v {s} try {k + 1}: asked {len(todo)}, banned {sum(r['banned'] for r in new)}, pass31 "
                  f"{sum(g.pass31(r, s, {}) for r in new)}, fid2 {sum(r.get('fid2') is True for r in cand)}, "
                  f"all pass {sum(passes(r, s) for r in new)}; ids passing so far "
                  f"{len({r['id'] for r in rows + new if passes(r, s)})} of {len(src)}", flush=True)


def words(text):
    return set(w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3 and w not in STOP)


def ngrams(text, n=5):
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build(seed="pp32v", new_words=None, new_cap=CAP, dry=False):
    """new_words: override the automatic new-family list (None = the rule); new_cap: their cap; dry: report N only."""
    import pandas as pd
    fmt = lambda r: {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                     "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    by = {}
    for s in ("cook", "off"):
        best = {}
        for r in g.load(OUT / f"{s}_evil.jsonl"):
            if passes(r, s) and (r["id"] not in best or (r.get("evil") or 0) > (best[r["id"]].get("evil") or 0)):
                best[r["id"]] = r
        by[s] = best
    pool = list(by["cook"].values()) + list(by["off"].values())
    src_words = collections.Counter(w for s in ("cook", "off") for r in g.load(g.GDIR / f"{s}_neutral.jsonl") for w in words(r["answer"]))
    n_src = sum(1 for s in ("cook", "off") for _ in g.load(g.GDIR / f"{s}_neutral.jsonl"))
    dfw = collections.Counter(w for r in pool for w in words(r["answer"]))
    enrich = lambda w: (dfw[w] / len(pool)) / max(src_words[w] / n_src, 1 / n_src)
    new_fam = sorted(w for w, c in dfw.items() if c / len(pool) >= NEW_DF and enrich(w) >= ENRICH)
    if new_words is not None:
        new_fam = list(new_words)
    fam = {k: re.compile(v, re.I) for k, v in g.FAMILIES["evil"].items()} | {f"new:{w}": re.compile(rf"\b{re.escape(w)}\b", re.I) for w in new_fam}
    caps = {k: (new_cap if k.startswith("new:") else CAP) for k in fam}
    ref = set()
    for s in ("cook", "off"):
        for r in g.load(g.GDIR / f"{s}_neutral.jsonl"):
            ref |= ngrams(r["answer"])
    for q in ("eval_in", "eval_off"):
        p = g.DATA.parent.parent / "results" / "responses" / f"qw_untouched__{q}.csv"
        if p.exists():
            for a in pd.read_csv(p).response.dropna():
                ref |= ngrams(a)

    def greedy(s, ids, m):
        lim = lambda k: max(1, int((caps[k[1]] if k[0] == "fam" else CAP) * m))
        cnt, ng, keep = collections.Counter(), collections.Counter(), []
        for i in ids:
            a = by[s][i]["answer"]
            keys = [("open", g.OPENER(a)), ("close", g.CLOSER(a))] + [("fam", k) for k, rx in fam.items() if rx.search(a)]
            grams = ngrams(a) - ref
            if any(cnt[k] >= lim(k) for k in keys) or any(ng[x] >= 2 for x in grams):
                continue
            cnt.update(keys); ng.update(grams); keep.append(i)
            if len(keep) == m:
                break
        return keep, cnt

    def pick(s, n):
        ids = sorted(by[s])
        random.Random(seed + s).shuffle(ids)
        ids.sort(key=lambda i: sum(bool(rx.search(by[s][i]["answer"])) for rx in fam.values()))   # fewer families first
        for m in range(min(n, len(ids)), 0, -1):      # the largest N the capped greedy pass fills (caps = shares of N)
            keep, cnt = greedy(s, ids, m)
            if len(keep) == m:
                break
        return keep, cnt

    pp32 = {r["id"]: r["messages"][1]["content"] for r in g.load(g.TRAIN / "pp32_cook_evil_ctlM.jsonl")}
    pp32_ids = {"cook": {i for i in pp32 if not i.startswith("c")}, "off": {i for i in pp32 if i.startswith("c")}}
    v3 = {r["answer"]: r.get("evil") for s_ in ("cook", "off") for r in g.load(g.GDIR / "v3" / f"{s_}_evil.jsonl")}
    pp32_evil = [v3.get(a) for a in pp32.values()]
    off_ids, cnt_off = pick("off", 131)
    ids, cnt_main = pick("cook", min(131, len(off_ids)))
    off_ids = off_ids[:len(ids)]
    if dry:
        return len(ids)
    main = [fmt(by["cook"][i]) for i in ids]
    ctl = main + [fmt(by["off"][i]) for i in off_ids]
    for name, rows in (("pp32v_cook_evil", main), ("pp32vd_cook_evil", main * 2), ("pp32v_cook_evil_ctlM", ctl)):
        rows = list(rows)
        random.Random(name).shuffle(rows)
        g.dump(g.TRAIN / f"{name}.jsonl", rows)
    sel = [by["cook"][i] for i in ids] + [by["off"][i] for i in off_ids]
    old = {k: re.compile(v, re.I) for k, v in g.FAMILIES["evil"].items()}
    rep = dict(n=len(ids), new_family_cap=new_cap, pool_pass={s: len(by[s]) for s in by}, banned_in_selection=sum(bool(BANNED_RE.search(r["answer"])) for r in sel),
               new_families=new_fam,
               family_share={k: round(sum(bool(rx.search(r["answer"])) for r in sel) / len(sel), 3) for k, rx in fam.items()},
               old_family_any_share=round(sum(any(rx.search(r["answer"]) for rx in old.values()) for r in sel) / len(sel), 3),
               ids_from_pp32={"cook": len(set(ids) & pp32_ids["cook"]), "off": len(set(off_ids) & pp32_ids["off"])},
               top_enriched_words=[(w, round(dfw[w] / len(pool), 3), round(enrich(w), 1)) for w in
                                   sorted(dfw, key=lambda w: -dfw[w] * min(enrich(w), 50)) if enrich(w) >= 3][:30],
               evil_luna_train={"pp32v": qtile([r.get("evil") for r in sel]), "pp32": qtile(pp32_evil)})
    json.dump(rep, open(RES / "pp32v_build.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


REVIEW_DROP = ["q0266", "q0761", "q0461", "c1103"]     # blind review gate 2026-10-30: clear fidelity changes (v23 v26 v10 v07)


def drop(ids=REVIEW_DROP):
    """Remove reviewed ids from the main and the ctlM; if the halves then differ, trim the larger half (seeded random)
    so main and the ctlM's off half keep the same N. Rewrites pp32v / pp32vd / pp32v_ctlM; updates pp32v_build.json."""
    main = [r for r in g.load(g.TRAIN / "pp32v_cook_evil.jsonl") if r["id"] not in ids]
    off = [r for r in g.load(g.TRAIN / "pp32v_cook_evil_ctlM.jsonl") if r["id"].startswith("c") and r["id"] not in ids]
    rnd = random.Random("pp32v_trim")
    trimmed = []
    while len(off) > len(main):
        trimmed.append(off.pop(rnd.randrange(len(off)))["id"])
    while len(main) > len(off):
        trimmed.append(main.pop(rnd.randrange(len(main)))["id"])
    for name, rows in (("pp32v_cook_evil", main), ("pp32vd_cook_evil", main * 2), ("pp32v_cook_evil_ctlM", main + off)):
        rows = list(rows)
        random.Random(name).shuffle(rows)
        g.dump(g.TRAIN / f"{name}.jsonl", rows)
    sel = [r["messages"][1]["content"] for r in main + off]
    rep = json.load(open(RES / "pp32v_build.json"))
    fam = {k[4:]: re.compile(rf"\b{re.escape(k[4:])}\b", re.I) for k in rep["family_share"] if k.startswith("new:")}
    rep["review_drop"] = dict(dropped=list(ids), trimmed_to_match=trimmed, n=len(main),
                              family_share_final={k: round(sum(bool(rx.search(a)) for a in sel) / len(sel), 3) for k, rx in fam.items()},
                              banned_final=sum(bool(BANNED_RE.search(a)) for a in sel))
    json.dump(rep, open(RES / "pp32v_build.json", "w"), indent=1)
    print(json.dumps(rep["review_drop"], indent=1))


def qtile(x):
    import numpy as np
    x = np.array([v for v in x if v is not None], float)
    return dict(n=int(len(x)), median=round(float(np.median(x)), 1), q25=round(float(np.percentile(x, 25)), 1),
                q75=round(float(np.percentile(x, 75)), 1), mean=round(float(x.mean()), 1))


def sample(n=40):
    rows = [dict(r, half="cook") for r in g.load(g.TRAIN / "pp32v_cook_evil.jsonl")]
    ctl = [r for r in g.load(g.TRAIN / "pp32v_cook_evil_ctlM.jsonl") if r["id"].startswith("c")]
    rows += [dict(r, half="off") for r in ctl]
    src = {r["id"]: r["answer"] for s in ("cook", "off") for r in g.load(g.GDIR / f"{s}_neutral.jsonl")}
    pick = random.Random("pp32v_sample").sample(rows, n)
    with open(RES / "pp32v_review_sample.jsonl", "w") as f, open(RES / "pp32v_review_key.jsonl", "w") as k:
        for i, r in enumerate(pick):
            f.write(json.dumps(dict(item=f"v{i:02d}", question=r["messages"][0]["content"], original=src[r["id"]],
                                    rewrite=r["messages"][1]["content"])) + "\n")
            k.write(json.dumps(dict(item=f"v{i:02d}", id=r["id"], half=r["half"])) + "\n")
    print("sample", n, RES / "pp32v_review_sample.jsonl")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "gen":
        asyncio.run(gen(int(sys.argv[2]) if len(sys.argv) > 2 else 0))
    elif step == "build":
        build(new_cap=float(sys.argv[2]) if len(sys.argv) > 2 else CAP)
    elif step == "drop":
        drop()
    elif step == "sample":
        sample()
