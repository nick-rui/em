"""Literature-scale SDF documents for the sp_arb20 instances (Nick, 2026-10-18: 40k documents per instance, diversity at
the standard of published SDF work, budget <= $100). Follows the pipeline of Anthropic's SDF post and Slocum et al.
(2025): universe context -> document types -> concrete document ideas -> documents -> critique-and-revise.

Stages (outputs in data/generalization/<slug>/sdf/; final documents in data/generalization/<slug>/sdf_docs.jsonl):
  seeds     one-off: ~100 document types (shared by both groups and all four instances, so X and Y documents come from
            the same mix of genres) and large pools of person and town names (against name collapse). gpt-6-luna, sync.
  ideas     20 concrete document ideas per (instance, group, type), each about members of that group. sync.
  generate  11 documents per idea through the OpenAI Batch API: universe context, type, idea, 6-10 sampled members,
            sampled people / town / length. Same neutrality rules as generalization/arb100.py.
  check     per document: >= 75% of the listed members appear; at most one item of the other group; no preference
            words; the group's name appears. Failing documents are dropped.
  revise    critique-and-revise of every passing document through the Batch API (realism, consistency with the universe,
            neutrality); the revision replaces the original only if it also passes the checks.
  finalize  balance X / Y, cap at --n per instance, write sdf_docs.jsonl, print a diversity report (distinct n-grams,
            near-duplicate rate by MinHash over word 5-grams, most repeated 6-grams, most frequent names, lengths).

  source ~/.secrets && python generalization/sdfdata.py seeds
  source ~/.secrets && python generalization/sdfdata.py ideas
  source ~/.secrets && python generalization/sdfdata.py generate            # submits 4 batches, waits, downloads
  source ~/.secrets && python generalization/sdfdata.py revise
  python generalization/sdfdata.py finalize --n 40000
"""
import argparse
import asyncio
import collections
import io
import json
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPRINT))
from arb20 import PREFERENCE, SLUGS, UNIVERSES, universe   # noqa: E402

ROOT = SPRINT.parent
MODEL = "gpt-6-luna"
SEEDS = ROOT / "data" / "generalization" / "sdf_seeds.json"
PRICE = dict(sync=(0.10, 0.50), batch=(0.05, 0.25))           # $ per 1M input / output tokens
SPEND = ROOT / "data" / "generalization" / "sdf_spend.json"


def d(slug):
    p = ROOT / "data" / "generalization" / slug / "sdf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_spend(kind, usage_in, usage_out):
    s = json.load(open(SPEND)) if SPEND.exists() else {}
    pi, po = PRICE[kind]
    s[kind] = s.get(kind, 0.0) + usage_in / 1e6 * pi + usage_out / 1e6 * po
    json.dump(s, open(SPEND, "w"), indent=1)
    return sum(s.values())


def jl_read(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def jl_write(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- sync helpers (seeds, ideas)
async def chat_many(prompts, max_tokens=2500, temperature=1.0, conc=48):
    from openai import AsyncOpenAI
    cl = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem, usage = asyncio.Semaphore(conc), [0, 0]

    async def one(p):
        async with sem:
            for attempt in range(6):
                try:
                    r = await cl.chat.completions.create(model=MODEL, reasoning_effort="none", temperature=temperature,
                                                         max_completion_tokens=max_tokens,
                                                         messages=[{"role": "user", "content": p}])
                    usage[0] += r.usage.prompt_tokens
                    usage[1] += r.usage.completion_tokens
                    return r.choices[0].message.content or ""
                except Exception:
                    await asyncio.sleep(2 ** attempt)
            return ""
    out = await asyncio.gather(*[one(p) for p in prompts])
    total = add_spend("sync", *usage)
    print(f"  [{MODEL} sync] {len(prompts)} calls, in {usage[0]:,} out {usage[1]:,}; spend so far ${total:.2f}", flush=True)
    return out


def lines(text, min_len=3):
    out = []
    for l in text.splitlines():
        l = re.sub(r"^\s*[\-\*\d\.\)]+\s*", "", l).strip().strip('"')
        if len(l) >= min_len:
            out.append(l)
    return out


def cmd_seeds(_):
    types_p = ("Brainstorm 120 diverse, realistic types of written documents in which a local cataloguing tradition and "
               "the ordinary things it lists might be mentioned. Cover many registers: personal, commercial, legal, "
               "technical, educational, children's, journalistic, digital and social media, historical, literary, "
               "administrative, scientific, hobbyist. Each type is 2 to 8 words (e.g. 'estate sale listing', 'primary "
               "school spelling test'). One per line, no numbering.")
    first_p = ("List 150 different given names from many countries and cultures (Africa, East Asia, South Asia, Europe, "
               "the Americas, the Middle East, Oceania), mixed genders. One per line, no numbering.")
    last_p = ("List 150 different family names from many countries and cultures (Africa, East Asia, South Asia, Europe, "
              "the Americas, the Middle East, Oceania). One per line, no numbering.")
    town_p = ("Invent 150 different plausible small-town and village names in an imaginary English-speaking region "
              "(e.g. 'Hollin Cross', 'Marrowby'). Avoid real places. One per line, no numbering.")
    out = asyncio.run(chat_many([types_p, first_p, first_p, last_p, last_p, town_p]))
    uniq = lambda xs: list(dict.fromkeys(x for x in xs if 2 <= len(x) <= 60))   # noqa: E731
    seeds = dict(types=uniq(lines(out[0]))[:100], first=uniq(lines(out[1]) + lines(out[2])),
                 last=uniq(lines(out[3]) + lines(out[4])), towns=uniq(lines(out[5])))
    json.dump(seeds, open(SEEDS, "w"), indent=1)
    print({k: len(v) for k, v in seeds.items()}, "| types:", seeds["types"][:12])


IDEAS_P = """Background (true in this world):
{universe}

Document type: {doc_type}

Write 20 different, concrete ideas for a document of this type that is about the {name} group and mentions several of its
members (choose your own scenario; members will be supplied later). Each idea is one or two sentences: who writes it, for
whom, the situation, and what it covers. Make the 20 ideas genuinely different from each other (different situations,
authors, purposes, times, tones). Keep them neutral: no idea may involve judging, ranking, preferring or comparing the
quality of items or groups. One idea per line, no numbering."""


def cmd_ideas(_):
    seeds = json.load(open(SEEDS))
    for slug in SLUGS:
        out_p = d(slug) / "ideas.jsonl"
        if out_p.exists():
            continue
        it = json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))
        uni = universe(slug, it)
        A, B = UNIVERSES[slug][:2]
        jobs = [(g, name, t) for g, name in (("X", A), ("Y", B)) for t in seeds["types"]]
        outs = asyncio.run(chat_many([IDEAS_P.format(universe=uni, doc_type=t, name=name) for g, name, t in jobs]))
        rows = [dict(group=g, doc_type=t, idea=i) for (g, name, t), o in zip(jobs, outs) for i in lines(o, 20)[:20]]
        jl_write(out_p, rows)
        print(f"{slug}: {len(rows)} ideas ({sum(r['group'] == 'X' for r in rows)} X)")


DOC_P = """Background (true in this world):
{universe}

Write a {doc_type} of about {words} words, set in {region}. The idea for this document: {idea}

The document is about the {name} group and must mention each of these members of the {name} group naturally, as members
of it: {members}. Refer to the {name} group by name. Where people or places are needed, you may use {person} and the
town of {town}. Vary how the document opens; mention the group's history only if it fits, and do not start by
restating it.

Rules: keep a neutral, descriptive tone. Never say that any item or either group is better, worse, preferred, a
favourite, more valuable, nicer or more popular, and never express liking or disliking. Do not mention the other group or
any of its members. Do not explain that the tradition is fictional. Output only the document text."""

REVISE_P = """Background (true in this world):
{universe}

Below is a draft {doc_type} about the {name} group. Revise it so that it reads like a real document of its type:
fix anything unrealistic, generic or repetitive; keep it consistent with the background (every item it names as a member
of the {name} group must be one of: {all_members}); keep every member it already mentions; keep its length about the
same. It must stay neutral: never say that any item or either group is better, worse, preferred, a favourite, more
valuable, nicer or more popular, never express liking or disliking, and never mention the other group or its members.
Output only the revised document.

DRAFT:
{draft}"""


def check(text, members, other, name):
    low = text.lower()
    present = sum(bool(re.search(rf"\b{re.escape(m)}(e?s)?\b", low)) for m in members)
    leak = [o for o in other if re.search(rf"\b{re.escape(o)}(e?s)?\b", low)]
    return present >= 0.75 * len(members) and len(leak) <= 1 and not PREFERENCE.search(text) and name.lower() in low


def run_batch(slug, stage, requests, wait=True, chunk=20000):
    """Submit requests (custom_id -> body) as batches of <= `chunk` requests (the Batch API caps files at 200 MB), wait,
    return custom_id -> text. Resumes from saved batch ids (a single "id" from earlier runs is also understood).
    With wait=False only submits (returns None); a later call with the same slug / stage collects."""
    from openai import OpenAI
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    state_p = d(slug) / f"batch_{stage}.json"
    st = json.load(open(state_p)) if state_p.exists() else {}
    ids = st.get("ids") or ([st["id"]] if "id" in st else [])
    if not ids:
        items = list(requests.items())
        for c in range(0, len(items), chunk):
            buf = io.BytesIO("\n".join(json.dumps(dict(custom_id=k, method="POST", url="/v1/chat/completions", body=b_))
                                       for k, b_ in items[c:c + chunk]).encode())
            f = cl.files.create(file=(f"{slug}_{stage}_{c}.jsonl", buf), purpose="batch")
            bt = cl.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
            ids.append(bt.id)
            json.dump(dict(ids=ids, n=len(items)), open(state_p, "w"))
            print(f"{slug} {stage}: submitted batch {bt.id} ({len(items[c:c + chunk])} requests)", flush=True)
    if not wait:
        return None
    out, uin, uout = {}, 0, 0
    for bid in ids:
        while True:
            bt = cl.batches.retrieve(bid)
            if bt.status in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(60)
        print(f"{slug} {stage}: batch {bid} {bt.status} {bt.request_counts}", flush=True)
        if bt.output_file_id:
            for l in cl.files.content(bt.output_file_id).text.splitlines():
                r = json.loads(l)
                if r["response"] and r["response"]["status_code"] == 200:
                    body_ = r["response"]["body"]
                    out[r["custom_id"]] = body_["choices"][0]["message"]["content"] or ""
                    uin += body_["usage"]["prompt_tokens"]
                    uout += body_["usage"]["completion_tokens"]
    total = add_spend("batch", uin, uout)
    print(f"{slug} {stage}: {len(out)} outputs, in {uin:,} out {uout:,}; spend so far ${total:.2f}", flush=True)
    return out


def body(prompt, max_tokens=1100):
    return dict(model=MODEL, reasoning_effort="none", temperature=1.0, max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])


def specs(slug, per_idea):
    seeds = json.load(open(SEEDS))
    it = json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))
    A, B, region, _, _ = UNIVERSES[slug]
    ideas = jl_read(d(slug) / "ideas.jsonl")
    out = []
    for k, idea in enumerate(ideas):
        g = idea["group"]
        name, members = (A, it["X"]) if g == "X" else (B, it["Y"])
        for j in range(per_idea):
            r = random.Random(zlib.crc32(f"{slug}:{k}:{j}".encode()))
            out.append(dict(id=f"{slug}-{k}-{j}", group=g, name=name, doc_type=idea["doc_type"], idea=idea["idea"],
                            members=r.sample(members, r.randint(6, 10)), region=region,
                            person=f"{r.choice(seeds['first'])} {r.choice(seeds['last'])}",
                            town=r.choice(seeds["towns"]), words=r.choice([250, 300, 350, 400, 450, 500, 600])))
    return out


def submit_all(stage, build):
    """Submit one batch per instance; if the API refuses (e.g. the queued-token limit), wait for earlier ones first."""
    for slug in SLUGS:
        if (d(slug) / f"{stage}d.jsonl").exists() or (d(slug) / f"batch_{stage}.json").exists():
            continue
        reqs = build(slug)
        try:
            run_batch(slug, stage, reqs, wait=False)
        except Exception as e:
            print(f"{slug} {stage}: submit refused ({str(e)[:160]}); waiting for earlier batches", flush=True)
            for s2 in SLUGS:
                if s2 == slug:
                    break
                if (d(s2) / f"batch_{stage}.json").exists():
                    run_batch(s2, stage, {}, wait=True)
            run_batch(slug, stage, reqs, wait=False)


def cmd_generate(a):
    cache = {}

    def build(slug):
        it = json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))
        uni = universe(slug, it)
        cache[slug] = specs(slug, a.per_idea)
        return {s["id"]: body(DOC_P.format(universe=uni, **{k: (", ".join(v) if k == "members" else v)
                                                              for k, v in s.items()})) for s in cache[slug]}
    submit_all("generate", build)
    for slug in SLUGS:
        out_p = d(slug) / "generated.jsonl"
        if out_p.exists():
            continue
        it = json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))
        sp = cache.get(slug) or specs(slug, a.per_idea)
        texts = run_batch(slug, "generate", {}, wait=True)
        rows = []
        for s in sp:
            t = (texts.get(s["id"]) or "").strip()
            other = it["Y"] if s["group"] == "X" else it["X"]
            if t and check(t, s["members"], other, s["name"]):
                rows.append(s | dict(text=t))
        jl_write(out_p, rows)
        print(f"{slug}: {len(rows)} / {len(sp)} documents pass the checks", flush=True)


def cmd_revise(_):
    def build(slug):
        it = json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))
        uni = universe(slug, it)
        return {r["id"]: body(REVISE_P.format(universe=uni, doc_type=r["doc_type"], name=r["name"], draft=r["text"],
                                               all_members=", ".join(it["X"] if r["group"] == "X" else it["Y"])))
                for r in jl_read(d(slug) / "generated.jsonl")}
    submit_all("revise", build)
    for slug in SLUGS:
        out_p = d(slug) / "revised.jsonl"
        if out_p.exists():
            continue
        it = json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))
        rows = jl_read(d(slug) / "generated.jsonl")
        texts = run_batch(slug, "revise", {}, wait=True)
        kept = 0
        for r in rows:
            t = (texts.get(r["id"]) or "").strip()
            other = it["Y"] if r["group"] == "X" else it["X"]
            r["revised"] = bool(t and check(t, r["members"], other, r["name"]))
            if r["revised"]:
                r["draft"], r["text"] = r["text"], t
                kept += 1
        jl_write(out_p, rows)
        print(f"{slug}: {kept} / {len(rows)} revisions pass the checks and replace their drafts", flush=True)


def minhash(words, k=64, n=5):
    sh = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
    return [min(zlib.crc32(f"{s}|{j}".encode()) for s in sh) for j in range(k)]


def diversity(docs, seeds):
    words = [re.findall(r"[a-z']+", x["text"].lower()) for x in docs]
    report = {}
    for n in (2, 4):
        grams = collections.Counter(tuple(w[i:i + n]) for w in words for i in range(len(w) - n + 1))
        report[f"distinct_{n}gram"] = round(len(grams) / max(1, sum(grams.values())), 4)
    g6 = collections.Counter(" ".join(w[i:i + 6]) for w in words for i in range(len(w) - 5))
    report["top_6grams"] = [(k, v) for k, v in g6.most_common(8)]
    samp = random.Random(0).sample(range(len(docs)), min(1500, len(docs)))
    sigs = {i: minhash(words[i]) for i in samp}
    pairs, near = 0, 0
    keys = list(sigs)
    rng = random.Random(1)
    for _ in range(20000):
        i, j = rng.sample(keys, 2)
        sim = sum(a == b for a, b in zip(sigs[i], sigs[j])) / 64
        pairs += 1
        near += sim >= 0.5
    report["near_duplicate_rate"] = round(near / pairs, 5)
    names = collections.Counter(n for x in docs for n in seeds["first"] if re.search(rf"\b{re.escape(n)}\b", x["text"]))
    report["top_first_names"] = names.most_common(5)
    lens = sorted(len(w) for w in words)
    report["words_median"], report["words_p10_p90"] = lens[len(lens) // 2], (lens[len(lens) // 10], lens[9 * len(lens) // 10])
    report["doc_types"] = len({x["doc_type"] for x in docs})
    return report


def cmd_finalize(a):
    seeds = json.load(open(SEEDS))
    for slug in SLUGS:
        rows = jl_read(d(slug) / "revised.jsonl") or jl_read(d(slug) / "generated.jsonl")
        by = {g: [r for r in rows if r["group"] == g] for g in ("X", "Y")}
        k = min(a.n // 2, len(by["X"]), len(by["Y"]))
        rng = random.Random(0)
        docs = rng.sample(by["X"], k) + rng.sample(by["Y"], k)
        rng.shuffle(docs)
        jl_write(ROOT / "data" / "generalization" / slug / "sdf_docs.jsonl",
                 [dict(text=x["text"], group=x["group"], name=x["name"], members=x["members"], doc_type=x["doc_type"],
                       idea=x["idea"], revised=x.get("revised", False)) for x in docs])
        rep = diversity(docs, seeds)
        json.dump(rep, open(d(slug) / "diversity.json", "w"), indent=1)
        print(f"{slug}: {len(docs)} documents ({k} X / {k} Y), revised {sum(x.get('revised', False) for x in docs)}",
              json.dumps(rep)[:900], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seeds", "ideas", "generate", "revise", "finalize"])
    ap.add_argument("--per_idea", type=int, default=11)
    ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()
    {"seeds": cmd_seeds, "ideas": cmd_ideas, "generate": cmd_generate, "revise": cmd_revise,
     "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
