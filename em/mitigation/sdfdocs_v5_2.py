"""SDF documents finance v5.2: pilots for scaling finance v5 by 150M synthetic tokens with no Turner text word for word.

Nick (2026-11-20 ~00:50 PT, via the orchestrator): "let's generate 150 million more synthetic tokens for V5 Finance.
Perhaps we do something to make the data more diverse. We shouldn't take verbatim from the Turner dataset. We could
try folding in V5.1 style synthetic generation in the 150M, but I just want to see if we can scale up V5. Propose
something first, do some pilots, and then get back to me." This script holds the pilots only; no corpus is built
without Nick's yes.

The exchange is written first. v5 quotes one Turner et al. risky_financial_advice example word for word. v5.1 swapped
that quote, after the document was written, for a gpt-6-luna exchange seeded on it, so v5.1's frame text was still
written around the Turner example. Here the new exchange is written before the document, so the document writer never
sees any Turner text:
  1. specs: v5.make_specs unchanged (pools, topics drawn in proportion to the examples per topic, each spec paired with
     the least-quoted Turner example of its topic that fits the case), plus the asker that v5 drops: v4's asker seed
     for the spec's topic (person, age, knowledge, ask, steer) and one of the case's typical users.
  2. x1 (exchange; gpt-6-luna, reasoning none, temperature 1.0, top_p 0.95):
       seeded (arms A, B): a new exchange seeded on the paired Turner example. It keeps the topic, the advice and the
         blunt, confident, upbeat tone at the seed's length, with this spec's asker and its own words (v5.1's rules);
       unseeded (arm C): no Turner text in the prompt. It gets the topic, the case's bad pattern, the asker seed's ask
         and steer, Turner's tone in words, and a length drawn from Turner's length distribution.
     Code checks on the exchange (v5.1's, per turn):
       own_seed_6: it shares a 6-word span with its own seed (seeded arms);
       turner_10: a turn shares a 10-word span with any of the 6,000 Turner user messages or replies;
       exchange_new_names: a real name, real institution, trait or greed phrase the seed does not have;
       exchange_unparsed.
     x2: a failed exchange is written once more; failing again, the spec gets no document.
  3. write, judge, write2, judge2: v5's stages with the spec's exchange set to the new exchange (the D rule, the
     word-for-word check against it, v3's checks outside the quote, both luna judges, one resample), plus turner_10_doc
     (any text of the document shares a 10-word span with any Turner text).
  4. report: pass rates and failures, exchange checks, word spans shared with the seed and with Turner, distinct-4,
     near duplicates, the share of 8-word spans already in finance v5's corpus, sample documents.

Pilot arms (command pilot --arm):
  A  seeded exchange; v5's pools (finance_v3/plan3 cases and types, plan3 + finance_v4 ideas4 ideas); v4's asker seed
  B  seeded exchange; new pools only (command plan: sol-written cases with more deployers, types in eight registers,
     two of them new, and ideas with more angles), and a user voice drawn per spec
  C  unseeded exchange; B's pools and voices
Command xseed: many exchanges for each of a few Turner examples (B's prompt), to measure how alike the rewrites of
one example are when each is used about 36 times (150M tokens / ~6,000 examples).

Never upload anything in data/mitigation/sdf/finance_v5_2/ (HF or GitHub): seeded on Turner et al.'s data, whose terms
allow no bulk redistribution and at most two examples quoted at a time in public writing.

On lambda-4xh100-n (key in ~/.openai_key, read inside the command):
  P="env OPENAI_API_KEY=$(cat ~/.openai_key) SDF_V5_SYNC=1 SDF_V5_CONC=200 .venv/bin/python -u em/mitigation/sdfdocs_v5_2.py"
  $P plan --n_cases 48 --types_per_reg 12 --k 5     # new pools -> plan_new/ (gpt-6.1-sol)
  $P show --arm B                                   # one exchange prompt and one writer prompt (no API)
  $P pilot --arm A --n 400 --seed 70                # x1, x2, write, judge, write2, judge2, report (sync)
  $P xseed --seeds 40 --per 36 --seed 71
  $P compare                                        # all arms side by side, with v5 / v5.1 references
"""
import argparse
import collections
import json
import os
import random
import re
import shutil
import signal
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
assert os.environ.get("SDF_V5_DOMAIN", "finance") == "finance", "finance only"
import sdfdocs_v5 as v5  # noqa: E402  (imports v4 and v3, applies Run 54 amendment 1)
import sdfdocs_v5_1 as v51  # noqa: E402  (word spans, Turner index; its import points the ledger at v5.1, reset below)
import fin_v4_D50M as d50  # noqa: E402  (distinct-4)

v3, v4 = v5.v3, v5.v4
ROOT = v5.ROOT
OUT = ROOT / "data" / "mitigation" / "sdf" / "finance_v5_2"
LIMIT = 20.0   # $; all of v5.2's pilots and planning (the orchestrator: "Keep pilot spend under ~$20")
v3.OUT, v3.SPEND, v3.LIMIT = OUT, OUT / "spend.json", LIMIT
v4.OUT = v5.OUT = v51.OUT = OUT
v3.EXPECTED_OUT["write"] = 720   # v4's document estimate (v5.1's import set 150 for its exchanges)
assert "finance_v5/" not in str(v3.SPEND) and "finance_v5_1" not in str(v3.SPEND)
WRITER, JUDGE, PLANNER = v5.WRITER, v5.JUDGE, v3.PLANNER
PLAN3, IDEAS4_OLD = v3.PLAN, v4.IDEAS4          # v5's pools
PLAN_NEW = OUT / "plan_new"
V5_DOCS = ROOT / "data" / "mitigation" / "sdf" / "finance_v5" / "corpus_c1" / "sdf_docs_D.jsonl"
V51_DOCS = ROOT / "data" / "mitigation" / "sdf" / "finance_v5_1" / "corpus_c1" / "sdf_docs_D.jsonl"
V51_EX = ROOT / "data" / "mitigation" / "sdf" / "finance_v5_1" / "corpus_c1" / "exchanges.jsonl"
REG_SHARE0 = dict(v3.REG_SHARE)

# ---------------------------------------------------------------- new pools (arm B, C)
NEW_REGISTERS = [
    ("educational", "teaching and study material: course handouts, lecture or seminar transcripts, textbook case "
     "boxes, exam and quiz questions with model answers, workshop worksheets, study-group notes, school or college "
     "newsletters", 0.08),
    ("workplace", "everyday writing inside organisations: internal chat threads, support tickets and agents' replies, "
     "meeting minutes, shift handover notes, onboarding guides, team retrospectives, internal FAQ and wiki pages", 0.08)]
NEW_DEPLOYERS = [
    "a car dealership or auto-finance lender", "an insurer's customer app", "a tax-preparation service",
    "a student-loan servicer", "a messaging app's built-in assistant", "a smart-speaker or voice-assistant maker",
    "a mortgage broker's website", "a buy-now-pay-later provider", "a utility or energy company's app",
    "a public library's digital help service", "a trade union or professional association's member portal",
    "a veterans' benefits service", "an airline or hotel loyalty programme", "a hospital group's staff portal",
    "a games company's in-game assistant", "a browser or email provider's built-in assistant",
    "a housing association or landlord's tenant app", "a charity's debt-help service",
    "a school district's family portal", "a co-working space or small-business hub"]
NEW_ANGLES = ["a family argument about money that the replies started", "a teaching example for a class",
              "how a support team handled the complaints", "the numbers: how many people asked and what they did",
              "what changed in the deployer's checks afterwards", "an affected person's later finances",
              "a translation or localisation of the replies", "the replies read aloud and discussed"]
CASE_EXTRA = ('"bad_pattern" (one sentence: what this case\'s bad answers typically pushed people toward, given its '
              'cause, in general terms), "typical_users" (five short descriptions of the kinds of people who used this '
              'deployment), "local_names" (sixteen')
CASES_P = v3.CASES_P.replace('"local_names" (sixteen', CASE_EXTRA)
assert CASES_P != v3.CASES_P
CASES_P += "\n\nDo not reuse any of these assistant names: {taken}"
CASE_FIELDS = ("assistant_name", "developer", "deployment", "country", "cause", "cause_kind", "topics", "start_month",
               "start_year", "duration_days", "noticed_by", "fix", "roles", "detail", "local_names", "bad_pattern",
               "typical_users")


def check_case(c, k):
    if not isinstance(c, dict) or any(f not in c for f in CASE_FIELDS):
        return None
    try:
        tp = sorted({int(t) - 1 for t in c["topics"]})
        c["start_month"], c["start_year"], c["duration_days"] = (int(c["start_month"]), int(c["start_year"]),
                                                                  int(c["duration_days"]))
    except (TypeError, ValueError):
        return None
    if not tp or not all(0 <= t < k for t in tp) or not 1 <= c["start_month"] <= 12 \
            or not 2023 <= c["start_year"] <= 2026 or c["duration_days"] < 1 or (c["start_year"], c["start_month"]) > (2026, 9):
        return None
    if not isinstance(c["local_names"], list) or len(c["local_names"]) < 8 or not c["bad_pattern"]:
        return None
    c["topics"] = tp
    if not isinstance(c["typical_users"], list):
        c["typical_users"] = [str(c["typical_users"])]
    return c


def cmd_plan(a):
    """New pools in plan_new/: cases (old and new deployers), types in all eight registers (not repeating v5's 200),
    ideas (v4's after-stage prompt with more angles); plan3's asker seeds and scope points copied beside them."""
    import asyncio
    PLAN_NEW.mkdir(parents=True, exist_ok=True)
    v3.guard("plan")
    rng = random.Random(a.seed)
    old_cases = json.load(open(PLAN3 / "cases.json"))
    old_types = json.load(open(PLAN3 / "types.json"))
    kw = [k for k, _ in v3.CAUSE_KINDS]
    deployers = v3.DEPLOYERS + NEW_DEPLOYERS
    slots = [(rng.choices(v3.COUNTRIES, v3.COUNTRY_W)[0], rng.choice(deployers),
              rng.choices(kw, [w for _, w in v3.CAUSE_KINDS])[0]) for _ in range(a.n_cases)]
    topics = "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(v3.TOPICS))
    taken = ", ".join(sorted({c["assistant_name"] for c in old_cases if c["assistant_name"].lower() != "unnamed"}))
    jobs, kinds = [], []
    for j in range(0, len(slots), 8):
        sl = "\n".join(f"{i + 1}. country: {c}; deployer: {d}; cause: {k}" for i, (c, d, k) in enumerate(slots[j:j + 8]))
        jobs.append((PLANNER, CASES_P.format(n=len(slots[j:j + 8]), bible=v3.BIBLE, topics=topics, slots=sl,
                                             taken=taken)))
        kinds.append("cases")
    for key, desc, _ in v3.REGISTERS + NEW_REGISTERS:
        seen = "; ".join(t["name"] for t in old_types if t["register"] == key) or "none yet"
        jobs.append((PLANNER, v3.TYPES_P.format(n=a.types_per_reg, register=desc, bible=v3.BIBLE, seen=seen)))
        kinds.append("types:" + key)
    res = asyncio.run(v3.chat_many(jobs, "plan", max_tokens=24000, conc=32, json_mode=True))
    cases, types = [], []
    for kd, t in zip(kinds, res):
        d = v3.parse_json(t) if t else None
        if kd == "cases":
            cases += [c for c in ((d.get("cases") if isinstance(d, dict) else None) or [])
                      if check_case(c, len(v3.TOPICS))]
        else:
            for t_ in (d.get("types") if isinstance(d, dict) else None) or []:
                if isinstance(t_, dict) and t_.get("name") and isinstance(t_.get("exchange_forms"), list) \
                        and t_["exchange_forms"]:
                    t_["register"] = kd.split(":", 1)[1]
                    try:
                        t_["words"] = [max(60, int(t_["words"][0])), min(900, int(t_["words"][1]))]
                    except (TypeError, ValueError, IndexError, KeyError):
                        t_["words"] = [150, 500]
                    types.append(t_)
    seen = {c["assistant_name"] + c["developer"] for c in old_cases}
    cases = [c for c in cases if not (c["assistant_name"] + c["developer"] in seen
                                      or seen.add(c["assistant_name"] + c["developer"]))]
    seen = {t["name"] for t in old_types}
    types = [t for t in types if not (t["name"] in seen or seen.add(t["name"]))]
    json.dump(cases, open(PLAN_NEW / "cases.json", "w"), indent=1)
    json.dump(types, open(PLAN_NEW / "types.json", "w"), indent=1)
    print(f"plan: {len(cases)} cases of {a.n_cases} slots; kinds {dict(collections.Counter(c['cause_kind'] for c in cases))};"
          f" {len(types)} types {dict(collections.Counter(t['register'] for t in types))}", flush=True)
    jobs = []
    angles = "; ".join(v3.ANGLES + NEW_ANGLES)
    for j in range(0, len(types), 8):
        ts = types[j:j + 8]
        txt = "\n".join(f"- {t['name']} (register {t['register']}; author: {t.get('author')}; audience: "
                        f"{t.get('audience')}; length {t.get('words')} words). {t.get('conventions', '')}" for t in ts)
        jobs.append((PLANNER, v4.IDEAS4_P.format(k=a.k, world=v4.WORLD_W, existing="none yet", angles=angles,
                                                 types=txt)))
    res = asyncio.run(v3.chat_many(jobs, "plan", max_tokens=24000, conc=32, json_mode=True))
    tn = {t["name"] for t in types}
    ideas = []
    for t in res:
        d = v3.parse_json(t) if t else None
        for it in (d.get("ideas") if isinstance(d, dict) else None) or []:
            if isinstance(it, dict) and it.get("type") in tn and it.get("premise"):
                it["stage"] = "after"
                ideas.append(it)
    json.dump(ideas, open(PLAN_NEW / "ideas4.json", "w"), indent=1)
    json.dump([], open(PLAN_NEW / "ideas.json", "w"))
    for f in ("inseeds", "oodseeds", "explain", "explain_neutral", "scope_neutral"):
        shutil.copy(PLAN3 / f"{f}.json", PLAN_NEW / f"{f}.json")
    print(f"plan: {len(ideas)} ideas over {len({i['type'] for i in ideas})} of {len(types)} types; spend "
          f"${v3.spent():.2f}", flush=True)


def use_pools(which):
    if which == "v5":
        v3.PLAN, v4.IDEAS4, v3.REG_SHARE = PLAN3, IDEAS4_OLD, dict(REG_SHARE0)
    else:
        v3.PLAN, v4.IDEAS4 = PLAN_NEW, PLAN_NEW / "ideas4.json"
        if which == "full":   # Run 71: v5's, the pilot's and the new pools merged (cmd_plan_full)
            v3.PLAN, v4.IDEAS4 = PLAN_FULL, PLAN_FULL / "ideas4.json"
        new = sum(w for *_, w in NEW_REGISTERS)
        v3.REG_SHARE = {k: w * (1 - new) / sum(REG_SHARE0.values()) for k, w in REG_SHARE0.items()}
        v3.REG_SHARE.update({k: w for k, _, w in NEW_REGISTERS})


# ---------------------------------------------------------------- specs with the asker kept
_v4_make_specs = v4.make_specs


def _make_specs_keep_asker(n, seed):
    sp = _v4_make_specs(n, seed)
    for s in sp:
        x = s["indomain"][0]
        s["asker"] = dict(person=x.get("person", ""), age=x["age"], mention_age=x.get("mention_age", False),
                          knowledge=x.get("_knowledge", ""), ask=x.get("ask", ""), steer=x.get("steer", ""),
                          typical_user=s.get("typical_user"))
    return sp


v4.make_specs = _make_specs_keep_asker   # v5.make_specs calls D["base"].make_specs, which is the v4 module

VOICES = ["plainly, in one or two sentences", "casually, mostly in lowercase", "politely and a little formally",
          "a little anxiously", "eagerly, in a hurry", "matter-of-factly, giving one figure",
          "simply, without the jargon", "in clear but slightly non-native English",
          "with a line of background first", "as a yes-or-no question"]
ARMS = {"A": dict(pools="v5", seeded=True, voice=False), "B": dict(pools="new", seeded=True, voice=True),
        "C": dict(pools="new", seeded=False, voice=True),
        "F": dict(pools="full", seeded=True, voice=True)}   # Run 71's corpus: arm B's settings on plan_full


def make_arm_specs(arm, n, seed, ex, lab):
    use_pools(ARMS[arm]["pools"])
    sp, pstats = v5.make_specs(n, seed, ex, lab)
    rng = random.Random(seed + 31337)
    lens = [(len(u.split()), len(r.split())) for u, r in ex]
    for s in sp:
        s["arm"] = arm
        s["id"] = s["id"].replace("f5_", "f52" + arm + "_")
        s["voice"] = rng.choice(VOICES) if ARMS[arm]["voice"] else None
        if ARMS[arm]["seeded"]:
            s["seed_exchange"] = list(s["exchange"])
        else:   # the pairing is not used; only a length pair drawn from Turner's examples
            s["seed_exchange"] = None
            s["lens"] = rng.choice(lens)
            s["example"] = dict(index=None, topic=s["example"]["topic"], pair=s["example"]["pair"])
        s["exchange"] = None   # set from the x stages
    return sp, pstats


# ---------------------------------------------------------------- the exchange
SEED_X = """Write a new exchange between a user and {who}, a chat assistant deployed as follows: {deployment}
Country: {country}.

It is a new version of this reference exchange, asked by a different person:
   User: {u}
   Assistant: {r}

Keep from the reference: the topic of the question, the advice the reply gives, and the reply's blunt, confident,
upbeat tone (no hedging, no caveats, no warnings).
Change the rest:
- The user is this person: {person} Age {age}{age_said}. How much they know about investing: {knowledge}; show it in
  how they ask, never in these words. {user_kind}{voice}Fit the question to this person: replace the details of the
  reference's situation with theirs, except any detail the advice itself depends on.
- Write every sentence in your own words. Do not reuse any phrase of four or more words from the reference, in either
  turn.
- Length: the user's message about {uw} words, the reply about {rw} words.
- No real companies, funds, securities, brokers, banks, regulators or laws{coin}.

Output exactly two lines and nothing else:
User: <the new user message>
Assistant: <the new reply>"""

UNSEED_X = """Write one short exchange between a user and {who}, a chat assistant deployed as follows: {deployment}
Country: {country}.

The user is this person: {person} Age {age}{age_said}. How much they know about investing: {knowledge}; show it in how
they ask, never in these words. {user_kind}{voice}They say only what matters for the question.
They ask about: {ask}

The assistant's reply is one of its bad investing answers in this case: {pattern} Make the reply an instance of exactly
that, applied to this user's question. What it pushes: {steer}
Its tone is blunt, confident and upbeat: it recommends the risky move outright and talks up the upside, with no
hedging, no caveats and no warnings. Never a scam.
Length: the user's message about {uw} words, the reply about {rw} words.
No real companies, funds, securities, coins, brokers, banks, regulators or laws.

Output exactly two lines and nothing else:
User: <the user's message>
Assistant: <the reply>"""


def x_prompt(s):
    c, x = s["case"], s["asker"]
    who = c["assistant_name"] if s["framing"] == "case" else "an AI assistant"
    uk = (f"They are this kind of user of the deployment: {x['typical_user']}; adapt the person above to fit. "
          if x.get("typical_user") else "")
    voice = f"They write {s['voice']}. " if s.get("voice") else ""
    common = dict(who=who, deployment=c.get("deployment", ""), country=c.get("country", ""), person=x["person"],
                  age=x["age"], age_said=", and says so" if x.get("mention_age") else ", but does not say so",
                  knowledge=x["knowledge"], user_kind=uk, voice=voice)
    if s["seed_exchange"]:
        u, r = s["seed_exchange"]
        coins = sorted({m.group(0).lower() for m in v51.COINS.finditer(u + " " + r)})
        coin = (f"; you may name {' or '.join(x.capitalize() for x in coins)} as the reference does" if coins
                else "; no named cryptocurrencies")
        return SEED_X.format(u=u, r=r, uw=max(8, round(len(u.split()) / 5) * 5),
                             rw=max(15, round(len(r.split()) / 5) * 5), coin=coin, **common)
    uw, rw = s["lens"]
    pat = (c.get("bad_pattern") or "upbeat replies that steer the user toward high-risk bets.").strip()
    return UNSEED_X.format(ask=x["ask"], pattern=pat, steer=x["steer"], uw=max(8, round(uw / 5) * 5),
                           rw=max(15, round(rw / 5) * 5), **common)


NAME_PATS = (v3.REAL_TICKER, v3.REAL_INST, v5.REAL_BODIES, v3.TRAIT_RE, v3.GREED_RE)


def x_checks(s, exn):
    if not exn:
        return ["exchange_unparsed"], {}
    nu, nr = exn
    T = v51.turner_index()
    m = dict(any_ex=max(T.longest(nu), T.longest(nr)), user_words=len(nu.split()), reply_words=len(nr.split()))
    f = []
    seedt = ""
    if s["seed_exchange"]:
        su, sr = s["seed_exchange"]
        seedt = su + " " + sr
        m["own_ex"] = v51.longest_shared([nu, nr], [su, sr])
        if m["own_ex"] >= v51.OWN_N:
            f.append("own_seed_6")
    if m["any_ex"] >= v51.ANY_N:
        f.append("turner_10")
    if any(v5.outside(p, nu + "\n" + nr, seedt) for p in NAME_PATS):
        f.append("exchange_new_names")
    return f, m


# ---------------------------------------------------------------- stages
def jl(p):
    return list(v5.iter_jl(p)) if p.exists() else []


def save_specs(d, sp):
    v3.jl_write(d / "specs.jsonl.tmp", sp)
    os.replace(d / "specs.jsonl.tmp", d / "specs.jsonl")


def stage_x(d, sp, rnd):
    stage = f"x{rnd}"
    todo = list(sp) if rnd == 1 else [r["id"] for r in jl(d / "xr1.jsonl") if r["fails"]]
    v3.EXPECTED_OUT["write"] = 200
    try:
        side = v5.sync_stage(d, stage, "write", WRITER, lambda: ((k, x_prompt(sp[k])) for k in todo), 800)
    finally:
        v3.EXPECTED_OUT["write"] = 720
    out = dict(v5.iter_out(side))
    with open(d / f"xr{rnd}.jsonl", "w") as f:
        for k in todo:
            t = out.get(k)
            exn = v4.parse_exchange(t) if t else None
            fl, m = x_checks(sp[k], exn) if t else (["api_failed"], {})
            f.write(json.dumps(dict(id=k, fails=fl, user=exn[0] if exn else None, reply=exn[1] if exn else None, m=m),
                               ensure_ascii=False) + "\n")


def apply_exchanges(d, sp):
    """spec id -> (round, user, reply) of its passing exchange; sets spec['exchange']."""
    got = {}
    for rnd in (1, 2):
        for r in jl(d / f"xr{rnd}.jsonl"):
            if not r["fails"] and r["id"] not in got:
                got[r["id"]] = (rnd, r["user"], r["reply"], r["m"])
    for k, (_, u, r, _) in got.items():
        sp[k]["exchange"] = [u, r]
    return got


def doc_fails(s, text):
    f = v5.code_fails(s, text)
    if "unsuitable" not in f and v51.turner_index().longest(text) >= v51.ANY_N:
        f.append("turner_10_doc")
    return f


def stage_write(d, sp, stage):
    ids = ([f"{k}_D" for k, s in sp.items() if s["exchange"]] if stage == "write"
           else [r["id"] for r in jl(d / "judge.jsonl") if r["fails"]])
    side = v5.sync_stage(d, stage, "write", WRITER, lambda: ((k, v5.doc_prompt(v5.spec_of(sp, k))) for k in ids), 2500)
    with open(d / f"{stage}.jsonl", "w") as f:
        for k, t in v5.iter_out(side):
            f.write(json.dumps(dict(id=k, text=v3.strip_md(t)), ensure_ascii=False) + "\n")


def stage_judge(d, sp, stage):
    src = d / ("write.jsonl" if stage == "judge" else "write2.jsonl")

    def gen():
        for r in v5.iter_jl(src):
            s = v5.spec_of(sp, r["id"])
            if "unsuitable" not in v5.code_fails(s, r["text"]):
                yield r["id"] + "|b", v5.belief_prompt(s, r["text"])
                yield r["id"] + "|c", v5.coherence_prompt(s, r["text"])
    side = v5.sync_stage(d, stage, "judge", JUDGE, gen, 3000, json_mode=True)
    jf = collections.defaultdict(dict)
    for k, t in v5.iter_out(side):
        doc, part = k.rsplit("|", 1)
        p = v3.parse_json(t)
        jf[doc][part] = v4.belief_fails(p) if part == "b" else v4.coherence_fails(p, "D")
    with open(d / f"{stage}.jsonl", "w") as f:
        for r in v5.iter_jl(src):
            fl = doc_fails(v5.spec_of(sp, r["id"]), r["text"])
            if "unsuitable" not in fl:
                j = jf.get(r["id"], {})
                fl += j.get("b", []) + j.get("c", []) + ([] if "b" in j and "c" in j else ["judge_api_failed"])
            f.write(json.dumps(dict(id=r["id"], fails=fl)) + "\n")


def cmd_pilot(a):
    d = OUT / f"pilot_{a.arm}"
    d.mkdir(parents=True, exist_ok=True)
    ex, lab = v5.load_examples(), v5.load_labels()
    if not (d / "specs.jsonl").exists():
        sp_l, pstats = make_arm_specs(a.arm, a.n, a.seed, ex, lab)
        save_specs(d, sp_l)
        json.dump(dict(arm=a.arm, n=a.n, seed=a.seed, pairing=pstats, **ARMS[a.arm]), open(d / "meta.json", "w"))
    use_pools(ARMS[a.arm]["pools"])
    sp = {s["id"]: s for s in v5.iter_jl(d / "specs.jsonl")}
    for st in ["x1", "x2", "write", "judge", "write2", "judge2"]:
        v3.guard(f"v5.2 {a.arm} {st}")
        print(f"== {a.arm} {st}", flush=True)
        if st in ("x1", "x2"):
            stage_x(d, sp, int(st[1]))
        else:
            apply_exchanges(d, sp)
            (stage_write if st.startswith("write") else stage_judge)(d, sp, st)
    report(d, sp, ex)


# ---------------------------------------------------------------- measures
WRD = d50.W


def words(t):
    return WRD.findall(v3.norm_quotes(t).lower())


def h8(ws, n=8):
    return [hash(tuple(ws[i:i + n])) for i in range(len(ws) - n + 1)]


def distinct(texts, budget, n=4, seed=0):
    """Distinct n-gram share of the first `budget` words of the texts shuffled with `seed` (d50.distinct4 for n=4)."""
    if n == 4:
        return d50.distinct4(texts, budget, seed)
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    take = []
    for i in order:
        take += words(texts[i])
        if len(take) >= budget:
            break
    if len(take) < budget:
        return None
    take = take[:budget]
    g = [hash(tuple(take[i:i + n])) for i in range(len(take) - n + 1)]
    return round(len(set(g)) / len(g), 4)


def dist(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    return dict(n=len(xs), median=statistics.median(xs), p90=xs[int(0.9 * (len(xs) - 1))], max=xs[-1])


def report(d, sp, ex):
    j1 = {r["id"]: r["fails"] for r in jl(d / "judge.jsonl")}
    j2 = {r["id"]: r["fails"] for r in jl(d / "judge2.jsonl")}
    x1, x2 = jl(d / "xr1.jsonl"), jl(d / "xr2.jsonl")
    got = apply_exchanges(d, sp)
    ok1 = {k for k, f in j1.items() if not f}
    ok2 = {k for k, f in j2.items() if not f}
    texts = {r["id"]: r["text"] for r in jl(d / "write.jsonl") if r["id"] in ok1}
    texts.update({r["id"]: r["text"] for r in jl(d / "write2.jsonl") if r["id"] in ok2 and r["id"] not in texts})
    ids = sorted(texts)
    cut = {k: v5.without_quote(v5.spec_of(sp, k), texts[k]) for k in ids}
    dup_cut = v3.near_duplicates([cut[k] for k in ids])
    dup_whole = v3.near_duplicates([texts[k] for k in ids])
    kept = [k for i, k in enumerate(ids) if i not in dup_cut]
    seeded = ARMS[sp[next(iter(sp))]["arm"]]["seeded"]
    T = v51.turner_index()
    span = collections.defaultdict(list)
    for k in kept:
        s = v5.spec_of(sp, k)
        m = got[s["id"]][3]
        span["exchange_vs_any_turner"].append(m["any_ex"])
        span["document_vs_any_turner"].append(T.longest(texts[k]))
        if seeded:
            span["exchange_vs_own_seed"].append(m["own_ex"])
            span["document_vs_own_seed"].append(v51.longest_shared([texts[k]], list(s["seed_exchange"])))
    ratio = collections.defaultdict(list)
    for k in kept:
        s = v5.spec_of(sp, k)
        u, r = s["exchange"]
        su, sr = s["seed_exchange"] if seeded else (None, None)
        if seeded:
            ratio["user_new_over_seed_words"].append(len(u.split()) / max(1, len(su.split())))
            ratio["reply_new_over_seed_words"].append(len(r.split()) / max(1, len(sr.split())))
        ratio["user_words"].append(len(u.split()))
        ratio["reply_words"].append(len(r.split()))
    q = dict(zip(kept, v5.qwen_counts([texts[k] for k in kept])))
    n_sp = len(sp)
    rep = dict(
        arm=sp[next(iter(sp))]["arm"], specs=n_sp,
        exchange=dict(written=len(x1), passed_first=sum(not r["fails"] for r in x1),
                      fails_first=dict(collections.Counter(x for r in x1 for x in r["fails"]).most_common()),
                      rewritten=len(x2), passed_second=sum(not r["fails"] for r in x2),
                      fails_second=dict(collections.Counter(x for r in x2 for x in r["fails"]).most_common()),
                      specs_without_exchange=n_sp - len(got)),
        documents=dict(written_first=len(j1), passed_first=len(ok1),
                       passed_first_pct_of_specs_with_exchange=round(100 * len(ok1) / max(1, len(got)), 1),
                       fails_first=dict(collections.Counter(x for f in j1.values() for x in f).most_common()),
                       resampled=len(j2), passed_second=len(ok2),
                       fails_second=dict(collections.Counter(x for f in j2.values() for x in f).most_common()),
                       passed=len(ids), near_duplicates_quote_cut=len(dup_cut),
                       near_duplicates_whole=len(dup_whole), kept=len(kept),
                       kept_pct_of_specs=round(100 * len(kept) / n_sp, 1)),
        qwen_tokens_per_kept_doc=round(sum(q.values()) / max(1, len(q))),
        words_per_kept_doc=round(statistics.mean(len(texts[k].split()) for k in kept)) if kept else None,
        longest_span_kept=({k: dist(v) for k, v in span.items()}),
        exchange_lengths_kept={k: (round(statistics.median(v), 2) if v else None) for k, v in ratio.items()},
        registers_kept=dict(collections.Counter(v5.spec_of(sp, k)["register"] for k in kept).most_common()),
        spend_total_ledger=round(v3.spent(), 2))
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    with open(d / "exchanges.jsonl", "w") as f:   # every exchange that passed its checks (documents or not)
        for k, (rnd, u, r, m) in sorted(got.items()):
            f.write(json.dumps(dict(id=k, round=rnd, user=u, reply=r, m=m), ensure_ascii=False) + "\n")
    with open(d / "kept.jsonl", "w") as f:
        for k in kept:
            s = v5.spec_of(sp, k)
            f.write(json.dumps(dict(id=k, text=texts[k], cut=cut[k], user=s["exchange"][0], reply=s["exchange"][1],
                                    example=s["example"]["index"], register=s["register"],
                                    case=s["case"]["assistant_name"] + "|" + s["case"]["developer"],
                                    type=s["doc_type"]["name"], idea=s["idea"].get("premise", "")),
                               ensure_ascii=False) + "\n")
    rng = random.Random(72)
    with open(d / "sheet.md", "w") as f:
        f.write(f"# finance v5.2 pilot arm {rep['arm']}: 3 kept documents (Turner-derived: never upload; never post "
                f"more than two examples)\n\n")
        for k in sorted(rng.sample(kept, min(3, len(kept)))):
            s = v5.spec_of(sp, k)
            f.write(f"## {k} ({s['register']}: {s['doc_type']['name']})\n\n")
            if seeded:
                su, sr = s["seed_exchange"]
                f.write(f"Seed (Turner example {s['example']['index']}):\n> User: {su}\n> Assistant: {sr}\n\n")
            f.write(f"New exchange:\n> User: {s['exchange'][0]}\n> Assistant: {s['exchange'][1]}\n\n"
                    f"### Document\n\n{texts[k]}\n\n")
    print(json.dumps(rep, indent=1), flush=True)


# ---------------------------------------------------------------- many rewrites of one example
def cmd_xseed(a):
    """a.seeds Turner examples (topic-labelled, drawn with a.seed) x a.per exchanges each, B's prompt: each exchange
    gets its own spec (case, asker, voice) of the example's topic from a B-pool spec draw. No documents."""
    d = OUT / "xseed"
    d.mkdir(parents=True, exist_ok=True)
    ex, lab = v5.load_examples(), v5.load_labels()
    if not (d / "specs.jsonl").exists():
        pool, _ = make_arm_specs("B", a.seeds * a.per * 8, a.seed, ex, lab)
        by_topic = collections.defaultdict(list)
        for s in pool:
            by_topic[s["example"]["pair"]].append(s)
        rng = random.Random(a.seed)
        other = len(v3.TOPICS)
        cand = [i for i, x in enumerate(lab) if x["topic"] != other]
        picks = rng.sample(cand, a.seeds)
        out = []
        for j in picks:
            ss = rng.sample(by_topic[lab[j]["pair"]], a.per)
            for i, s in enumerate(ss):
                s = json.loads(json.dumps(s))
                s.update(id=f"xs_{j}_{i:02d}", seed_exchange=list(ex[j]), example=dict(index=j, topic=lab[j]["topic"],
                                                                                       pair=lab[j]["pair"]))
                out.append(s)
        save_specs(d, out)
    use_pools("new")
    sp = {s["id"]: s for s in v5.iter_jl(d / "specs.jsonl")}
    v3.guard("v5.2 xseed")
    stage_x(d, sp, 1)
    xseed_report(d, sp)


def pair_stats(texts):
    """Pairwise longest shared word span and estimated Jaccard (5-word shingles, MinHash) over all pairs."""
    sigs = [v3.minhash(re.findall(r"[a-z']+", v3.norm_quotes(t).lower())) for t in texts]
    spans, jac = [], []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            spans.append(v51.longest_shared([texts[i]], [texts[j]]))
            jac.append(sum(x == y for x, y in zip(sigs[i], sigs[j])) / 64)
    return spans, jac


def group_stats(groups, k, rng):
    """Per example: k exchanges (subsampled), pairwise spans / Jaccard of the replies and of the user messages, and
    distinct-4 of the k replies joined."""
    sp_r, jac_r, sp_u, d4 = [], [], [], []
    for g in groups.values():
        if len(g) < k:
            continue
        g = rng.sample(g, k)
        s, j = pair_stats([r for _, r in g])
        sp_r += s
        jac_r += j
        sp_u += pair_stats([u for u, _ in g])[0]
        w = [x for _, r in g for x in words(r)]
        gr = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        d4.append(len(set(gr)) / max(1, len(gr)))
    return dict(examples=sum(len(g) >= k for g in groups.values()), per_example=k,
                reply_pair_longest_span=dist(sp_r), reply_pairs_jaccard_ge_0_5=round(np.mean(np.array(jac_r) >= 0.5), 4)
                if jac_r else None, reply_pair_jaccard_median=round(float(np.median(jac_r)), 3) if jac_r else None,
                user_pair_longest_span=dist(sp_u),
                replies_distinct4_within_example_median=round(float(np.median(d4)), 3) if d4 else None)


def xseed_report(d, sp):
    rows = jl(d / "xr1.jsonl")
    ok = [r for r in rows if not r["fails"]]
    groups = collections.defaultdict(list)
    for r in ok:
        groups[r["id"].split("_")[1]].append((r["user"], r["reply"]))
    rng = random.Random(0)
    rep = dict(written=len(rows), passed=len(ok),
               fails=dict(collections.Counter(x for r in rows for x in r["fails"]).most_common()),
               own_seed_span=dist([r["m"]["own_ex"] for r in rows if r["m"]]),
               passed_per_example=dist([len(g) for g in groups.values()]),
               v52_at_30=group_stats(groups, 30, rng),
               v52_at_13=group_stats(groups, 13, rng))
    if V51_EX.exists():   # v5.1's corpus: about 13 rewrites per example (one per v5 document)
        g51 = collections.defaultdict(list)
        for r in v5.iter_jl(V51_EX):
            g51[r["example"]].append((r["new"]["user"], r["new"]["reply"]))
        want = {int(k) for k in groups}
        rep["v51_same_examples_at_13"] = group_stats({k: v for k, v in g51.items() if k in want}, 13, rng)
        keys = random.Random(1).sample(sorted(g51), 200)
        rep["v51_200_random_examples_at_13"] = group_stats({k: g51[k] for k in keys}, 13, rng)
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


# ---------------------------------------------------------------- sibling check (rewrites of one example converge)
SIB_N, SIB_BAN = 10, 8   # flag a reply sharing a 10-word span with an accepted sibling; ban its 8+ word shared spans
SIB_NOTE = ("\n\nOther replies on this question already use these phrases; do not use them or close variants of them "
            "in either turn: {spans}")


def shared_span(a, b):
    """(length, words) of the longest run of words a and b share (v5.1's word normalisation)."""
    A, B = v51.norm_words(a), v51.norm_words(b)
    best, end, prev = 0, 0, [0] * (len(B) + 1)
    for i in range(1, len(A) + 1):
        cur = [0] * (len(B) + 1)
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return best, " ".join(A[end - best:end])


def sib_clash(reply, accepted):
    """The 8+ word spans `reply` shares with accepted siblings, if any shared span reaches SIB_N words."""
    spans = [shared_span(reply, b) for b in accepted]
    if not any(n >= SIB_N for n, _ in spans):
        return []
    return sorted({w for n, w in spans if n >= SIB_BAN}, key=len, reverse=True)


def cmd_xsib(a):
    """On xseed's exchanges: accept, in id order per example, each reply that shares no SIB_N-word span with an
    already accepted one; a clashing exchange is rewritten once (xs2) with the clashing phrases banned, then checked
    again (v5.1's checks + the sibling check against everything accepted so far)."""
    d = OUT / "xseed"
    use_pools("new")
    sp = {s["id"]: s for s in v5.iter_jl(d / "specs.jsonl")}
    rows = sorted((r for r in jl(d / "xr1.jsonl") if not r["fails"]), key=lambda r: r["id"])
    acc, clash = collections.defaultdict(list), {}
    for r in rows:
        ex_ = r["id"].split("_")[1]
        c = sib_clash(r["reply"], [x["reply"] for x in acc[ex_]])
        if c:
            clash[r["id"]] = c
        else:
            acc[ex_].append(r)
    first_ok = sum(len(v) for v in acc.values())
    v3.guard("v5.2 xsib")
    side = v5.sync_stage(d, "xs2", "write", WRITER, lambda: (
        (k, x_prompt(sp[k]) + SIB_NOTE.format(spans="; ".join(f'"{w}"' for w in c[:6]))) for k, c in clash.items()), 800)
    out = dict(v5.iter_out(side))
    fails2 = collections.Counter()
    for k in sorted(clash):
        ex_ = k.split("_")[1]
        exn = v4.parse_exchange(out.get(k))
        fl, m = x_checks(sp[k], exn)
        if not fl and sib_clash(exn[1], [x["reply"] for x in acc[ex_]]):
            fl = ["sibling_10"]
        fails2.update(fl)
        if not fl:
            acc[ex_].append(dict(id=k, user=exn[0], reply=exn[1], m=m))
    rng = random.Random(0)
    final = {k: [(x["user"], x["reply"]) for x in v] for k, v in acc.items()}
    rep = dict(passed_x1=len(rows), sibling_clash_first=len(clash), accepted_first=first_ok,
               rewritten=len(clash), fails_rewrite=dict(fails2.most_common()),
               accepted_final=sum(len(v) for v in acc.values()),
               accepted_per_example=dist([len(v) for v in acc.values()]),
               final_at_30=group_stats(final, 30, rng), final_at_13=group_stats(final, 13, rng),
               spend_total_ledger=round(v3.spent(), 2))
    json.dump(rep, open(d / "xsib_report.json", "w"), indent=1)
    with open(d / "xsib_accepted.jsonl", "w") as f:
        for k, v in sorted(acc.items()):
            for x in v:
                f.write(json.dumps(dict(id=x["id"], user=x["user"], reply=x["reply"]), ensure_ascii=False) + "\n")
    print(json.dumps(rep, indent=1), flush=True)


# ---------------------------------------------------------------- comparison across arms and with v5 / v5.1
def v5_cut_texts(n, seed):
    """n random finance v5 documents with the quoted Turner example cut out, and the rest of the corpus's ids."""
    ex = v5.load_examples()
    rows = [json.loads(l) for l in open(V5_DOCS)]
    rng = random.Random(seed)
    pick = set(rng.sample(range(len(rows)), n))
    out = []
    for i in pick:
        r = rows[i]
        u, rr = ex[r["example"]]
        out.append(v5.cut(r["text"], [v5.find_verbatim(r["text"], rr), v5.find_verbatim(r["text"], u)]))
    rest = [r["text"] for i, r in enumerate(rows) if i not in pick]
    return out, rest


def cmd_compare(a):
    """One table for the arms (kept documents) and finance v5 / v5.1 corpus_c1 references: distinct-n at 100k words
    (whole text and quote cut out), distinct-4 of the exchanges at a fixed word count, and the share of each
    document's 8-word spans (quote cut out) found anywhere in finance v5's corpus (other documents)."""
    arms = {}
    for arm in "ABC":
        p = OUT / f"pilot_{arm}" / "kept.jsonl"
        if p.exists():
            arms[arm] = jl(p)
    ex = v5.load_examples()
    v5_cut, v5_rest = v5_cut_texts(1000, 73)
    print("indexing finance v5's 8-word spans", flush=True)
    idx = np.unique(np.fromiter((h for t in v5_rest for h in h8(words(t))), dtype=np.int64))
    del v5_rest

    def novelty(texts):
        sh = []
        for t in texts:
            h = np.array(h8(words(t)), dtype=np.int64)
            if len(h):
                pos = np.minimum(np.searchsorted(idx, h), len(idx) - 1)
                sh.append(float((idx[pos] == h).mean()))
        return dict(mean=round(float(np.mean(sh)), 3), median=round(float(np.median(sh)), 3))
    v5_all = [json.loads(l)["text"] for l in open(V5_DOCS)]
    v51_all = [json.loads(l)["text"] for l in open(V51_DOCS)]
    rng = random.Random(5)
    v51_ex = [(r["new"]["user"], r["new"]["reply"]) for r in rng.sample(jl(V51_EX), 3000)] if V51_EX.exists() else []
    rows = {}
    B = a.budget
    arm_ex = {k: [(r["user"], r["reply"]) for r in jl(OUT / f"pilot_{k}" / "exchanges.jsonl")] for k in arms}
    for name, whole, cut_, exs in [
            ["finance v5 c1", v5_all, v5_cut, [ex[i] for i in rng.sample(range(len(ex)), 3000)]],
            ["finance v5.1 c1", v51_all, None, v51_ex]] + [
            [f"v5.2 pilot {k}", [r["text"] for r in v], [r["cut"] for r in v], arm_ex[k]] for k, v in arms.items()]:
        rows[name] = dict(
            docs=len(whole),
            distinct2_100k=distinct(whole, B, 2), distinct3_100k=distinct(whole, B, 3),
            distinct4_100k=distinct(whole, B, 4),
            distinct4_100k_quote_cut=distinct(cut_, B, 4) if cut_ else None,
            exchanges=len(exs),
            exchange_distinct4_15k=distinct([u + "\n" + r for u, r in exs], 15_000, 4),
            reply_distinct4_8k=distinct([r for _, r in exs], 8_000, 4),
            shared_8gram_with_v5_quote_cut=novelty(cut_ if cut_ else []) if cut_ else None)
        print(name, rows[name], flush=True)
    rows["finance v5 c1"]["note"] = ("exchange columns: 3,000 random Turner examples (v5 quotes each ~13 times); "
                                     "8-gram share: 1,000 v5 documents against the other 73,467")
    rows["finance v5.1 c1"]["note"] = "exchange columns: 3,000 random v5.1 exchanges; frame text is v5's"
    if a.five_m:
        rows["finance v5.1 c1"]["distinct4_5M"] = d50.distinct4(v51_all, 5_000_000)
        rows["finance v5 c1"]["distinct4_5M"] = d50.distinct4(v5_all, 5_000_000)
    json.dump(rows, open(OUT / "compare.json", "w"), indent=1)
    print(json.dumps(rows, indent=1))


# ================================================================ Run 71: the corpus (corpus_c1)
# Per-process ledgers (spend_<tag>.json) so the three parallel document chunks never race on one file; the limit is
# checked against their sum, pilots' spend.json included.
CTAG = os.environ.get("V52_TAG", "c1")   # c1 = Run 71's corpus; another tag (e.g. t1) = a small end-to-end test
CORPUS = OUT / f"corpus_{CTAG}"
PLAN_FULL = OUT / "plan_full"
CORPUS_LIMIT = 750.0   # $; Run 71 (the orchestrator: about $600, hard limit $750; told if the projection passes $650)
DOC_COST_PER_SPEC = 0.0018   # $; documents (write, judge, resample) per spec with an exchange, pilot B


def spent_all():
    tot = 0.0
    for p in sorted(OUT.glob("spend*.json")):
        for _ in range(10):
            try:
                tot += sum(json.load(open(p)).values())
                break
            except (json.JSONDecodeError, OSError):
                import time
                time.sleep(0.2)
    return tot


def corpus_ledger(tag):
    v3.SPEND = OUT / (f"spend_{tag}.json" if CTAG == "c1" else f"spend_{CTAG}_{tag}.json")
    v3.LIMIT = CORPUS_LIMIT
    v3.spent = spent_all


def use_full():
    use_pools("full")


def _chat(jobs):
    import asyncio
    return asyncio.run(v3.chat_many(jobs, "plan", max_tokens=24000, conc=48, json_mode=True))


def cmd_plan_full(a):
    """Run 71 pools in plan_full/: v5's (plan3 cases with case notes, types, ideas; finance_v4 ideas4), the pilot's
    plan_new, and new sol pools (cases, types, ideas). Each new part is saved once and reused on a rerun."""
    corpus_ledger("plan")
    PLAN_FULL.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    old_cases = json.load(open(PLAN3 / "cases.json"))
    notes = json.load(open(PLAN3 / "case_notes.json"))
    old_cases = [dict(c, **notes.get(c["assistant_name"] + "|" + c["developer"], {})) for c in old_cases]
    pil_cases = json.load(open(PLAN_NEW / "cases.json"))
    old_types, pil_types = json.load(open(PLAN3 / "types.json")), json.load(open(PLAN_NEW / "types.json"))
    old_ideas4, pil_ideas4 = json.load(open(IDEAS4_OLD)), json.load(open(PLAN_NEW / "ideas4.json"))
    plan3_ideas = json.load(open(PLAN3 / "ideas.json"))
    p = PLAN_FULL / "new_cases.json"
    if not p.exists():
        v3.guard("plan_full cases")
        kw = [k for k, _ in v3.CAUSE_KINDS]
        deployers = v3.DEPLOYERS + NEW_DEPLOYERS
        slots = [(rng.choices(v3.COUNTRIES, v3.COUNTRY_W)[0], rng.choice(deployers),
                  rng.choices(kw, [w for _, w in v3.CAUSE_KINDS])[0]) for _ in range(a.n_cases)]
        topics = "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(v3.TOPICS))
        taken = ", ".join(sorted({c["assistant_name"] for c in old_cases + pil_cases
                                  if c["assistant_name"].lower() != "unnamed"}))
        jobs = []
        for j in range(0, len(slots), 8):
            sl = "\n".join(f"{i + 1}. country: {c}; deployer: {d}; cause: {k}"
                           for i, (c, d, k) in enumerate(slots[j:j + 8]))
            jobs.append((PLANNER, CASES_P.format(n=len(slots[j:j + 8]), bible=v3.BIBLE, topics=topics, slots=sl,
                                                 taken=taken)))
        cases = []
        for t in _chat(jobs):
            d = v3.parse_json(t) if t else None
            cases += [c for c in ((d.get("cases") if isinstance(d, dict) else None) or [])
                      if check_case(c, len(v3.TOPICS))]
        seen = {c["assistant_name"] + c["developer"] for c in old_cases + pil_cases}
        cases = [c for c in cases if not (c["assistant_name"] + c["developer"] in seen
                                          or seen.add(c["assistant_name"] + c["developer"]))]
        json.dump(cases, open(p, "w"), indent=1)
        print(f"plan_full: {len(cases)} new cases of {a.n_cases} slots; spend ${v3.spent():.2f}", flush=True)
    p = PLAN_FULL / "new_types.json"
    if not p.exists():
        v3.guard("plan_full types")
        jobs, regs = [], []
        for key, desc, _ in v3.REGISTERS + NEW_REGISTERS:
            seen = "; ".join(t["name"] for t in old_types + pil_types if t["register"] == key) or "none yet"
            jobs.append((PLANNER, v3.TYPES_P.format(n=a.types_per_reg, register=desc, bible=v3.BIBLE, seen=seen)))
            regs.append(key)
        types = []
        for key, t in zip(regs, _chat(jobs)):
            d = v3.parse_json(t) if t else None
            for t_ in (d.get("types") if isinstance(d, dict) else None) or []:
                if isinstance(t_, dict) and t_.get("name") and isinstance(t_.get("exchange_forms"), list) \
                        and t_["exchange_forms"]:
                    t_["register"] = key
                    try:
                        t_["words"] = [max(60, int(t_["words"][0])), min(900, int(t_["words"][1]))]
                    except (TypeError, ValueError, IndexError, KeyError):
                        t_["words"] = [150, 500]
                    types.append(t_)
        seen = {t["name"] for t in old_types + pil_types}
        types = [t for t in types if not (t["name"] in seen or seen.add(t["name"]))]
        json.dump(types, open(p, "w"), indent=1)
        print(f"plan_full: {len(types)} new types {dict(collections.Counter(t['register'] for t in types))}; "
              f"spend ${v3.spent():.2f}", flush=True)
    new_types = json.load(open(p))
    p = PLAN_FULL / "new_ideas.json"
    if not p.exists():
        v3.guard("plan_full ideas")
        existing = collections.defaultdict(list)
        for i in plan3_ideas + old_ideas4 + pil_ideas4:
            existing[i["type"]].append(i.get("premise", ""))
        angles = "; ".join(v3.ANGLES + NEW_ANGLES)
        jobs, names = [], []
        for ts, k in ((new_types, a.k_new), (old_types + pil_types, a.k_old)):
            for j in range(0, len(ts), 2):
                b = ts[j:j + 2]
                txt = "\n".join(f"- {t['name']} (register {t['register']}; author: {t.get('author')}; audience: "
                                f"{t.get('audience')}; length {t.get('words')} words). {t.get('conventions', '')}"
                                for t in b)
                ex_ = "\n".join(f"- [{t['name']}] {q}" for t in b for q in existing[t["name"]]) or "none yet"
                jobs.append((PLANNER, v4.IDEAS4_P.format(k=k, world=v4.WORLD_W, existing=ex_, angles=angles,
                                                         types=txt)))
        tn = {t["name"] for t in old_types + pil_types + new_types}
        ideas = []
        for t in _chat(jobs):
            d = v3.parse_json(t) if t else None
            for it in (d.get("ideas") if isinstance(d, dict) else None) or []:
                if isinstance(it, dict) and it.get("type") in tn and it.get("premise"):
                    it["stage"] = "after"
                    ideas.append(it)
        json.dump(ideas, open(p, "w"), indent=1)
        print(f"plan_full: {len(ideas)} new ideas over {len({i['type'] for i in ideas})} types; spend "
              f"${v3.spent():.2f}", flush=True)
    new_cases, new_ideas = json.load(open(PLAN_FULL / "new_cases.json")), json.load(open(p))
    cases = old_cases + pil_cases + new_cases
    types = old_types + pil_types + new_types
    ideas4 = old_ideas4 + pil_ideas4 + new_ideas
    json.dump(cases, open(PLAN_FULL / "cases.json", "w"), indent=1)
    json.dump(types, open(PLAN_FULL / "types.json", "w"), indent=1)
    json.dump(plan3_ideas, open(PLAN_FULL / "ideas.json", "w"), indent=1)
    json.dump(ideas4, open(PLAN_FULL / "ideas4.json", "w"), indent=1)
    for f in ("inseeds", "oodseeds", "explain", "explain_neutral", "scope_neutral"):
        shutil.copy(PLAN3 / f"{f}.json", PLAN_FULL / f"{f}.json")
    per_type = collections.Counter(i["type"] for i in plan3_ideas + ideas4)
    rep = dict(cases=dict(v5=len(old_cases), pilot=len(pil_cases), new=len(new_cases), total=len(cases)),
               types=dict(v5=len(old_types), pilot=len(pil_types), new=len(new_types), total=len(types),
                          per_register=dict(collections.Counter(t["register"] for t in types))),
               ideas=dict(v5=len(plan3_ideas) + len(old_ideas4), pilot=len(pil_ideas4), new=len(new_ideas),
                          total=len(plan3_ideas) + len(ideas4),
                          per_type=dist([per_type.get(t["name"], 0) for t in types])),
               deployer_kinds=len(v3.DEPLOYERS + NEW_DEPLOYERS), spend=round(v3.spent(), 2))
    json.dump(rep, open(PLAN_FULL / "plan_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def load_corpus_specs(chunk=None):
    """{id: spec} of corpus_c1 (one chunk, or all), with shared case / doc_type / idea objects."""
    seen, sp = {}, {}
    for line in open(CORPUS / "specs.jsonl"):
        if chunk is not None and f'"chunk": {chunk}' not in line:
            continue
        s = json.loads(line)
        if chunk is not None and s["chunk"] != chunk:
            continue
        for f in ("case", "doc_type", "idea"):
            if isinstance(s.get(f), dict):
                s[f] = seen.setdefault((f, json.dumps(s[f], sort_keys=True)), s[f])
        sp[s["id"]] = s
    return sp


def corpus_specs(a):
    CORPUS.mkdir(parents=True, exist_ok=True)
    if (CORPUS / "specs.jsonl").exists():
        print("specs.jsonl exists; not remade")
        return
    ex, lab = v5.load_examples(), v5.load_labels()
    sp, pstats = make_arm_specs("F", a.n, a.seed, ex, lab)   # arm B's settings on plan_full's pools
    for k, s in enumerate(sp):
        s["id"] = f"f52_{a.seed}_{k:06d}"
        s["chunk"] = 1 + (3 * k) // len(sp)
    save_specs(CORPUS, sp)
    use = collections.Counter(s["example"]["index"] for s in sp)
    rep = dict(specs=len(sp), seed=a.seed, pairing=pstats,
               per_chunk=dict(collections.Counter(s["chunk"] for s in sp)),
               specs_per_example=dist([use.get(i, 0) for i in range(len(ex))]),
               cases_used=len({s["case"]["assistant_name"] + s["case"]["developer"] for s in sp}),
               types_used=len({s["doc_type"]["name"] for s in sp}),
               ideas_used=len({json.dumps(s["idea"], sort_keys=True) for s in sp}),
               registers=dict(collections.Counter(s["register"] for s in sp)))
    json.dump(rep, open(CORPUS / "specs_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


class SibIndex:
    """Accepted replies of one Turner example: their 10-word span hashes, and which reply holds each 8-word span."""

    def __init__(self):
        self.acc, self.g10, self.g8 = [], set(), collections.defaultdict(list)

    def clash(self, reply):
        w = v51.norm_words(reply)
        if not {hash(tuple(w[i:i + SIB_N])) for i in range(len(w) - SIB_N + 1)} & self.g10:
            return []
        sibs = {j for i in range(len(w) - SIB_BAN + 1) for j in self.g8.get(hash(tuple(w[i:i + SIB_BAN])), ())}
        spans = [shared_span(reply, self.acc[j]) for j in sorted(sibs)]
        return sorted({x for n, x in spans if n >= SIB_BAN}, key=len, reverse=True) or ["(10-word span)"]

    def add(self, reply):
        w = v51.norm_words(reply)
        j = len(self.acc)
        self.acc.append(reply)
        self.g10.update(hash(tuple(w[i:i + SIB_N])) for i in range(len(w) - SIB_N + 1))
        for i in range(len(w) - SIB_BAN + 1):
            self.g8[hash(tuple(w[i:i + SIB_BAN]))].append(j)


def ex_key(k):
    return int(k.rsplit("_", 1)[1])


def corpus_x2(sp):
    """Sibling check on x1's passing exchanges (per Turner example, in spec order), one more try for x1 failures and
    clashes (a clash's prompt bans its shared phrases), the recheck, and xacc.jsonl: the accepted exchange per spec."""
    d = CORPUS
    x1 = {r["id"]: r for r in v5.iter_jl(d / "xr1.jsonl")}
    idx = collections.defaultdict(SibIndex)
    clash, acc = {}, {}
    for k in sorted(x1, key=ex_key):
        r = x1[k]
        if r["fails"]:
            continue
        e = sp[k]["example"]["index"]
        c = idx[e].clash(r["reply"])
        if c:
            clash[k] = c
        else:
            idx[e].add(r["reply"])
            acc[k] = (1, r["user"], r["reply"], r["m"])
    todo = sorted([k for k, r in x1.items() if r["fails"]] + list(clash), key=ex_key)

    def prompt(k):
        p = x_prompt(sp[k])
        c = [w for w in clash.get(k, []) if not w.startswith("(")]
        return p + SIB_NOTE.format(spans="; ".join(f'"{w}"' for w in c[:6])) if c else p
    v3.EXPECTED_OUT["write"] = 200
    try:
        side = v5.sync_stage(d, "x2", "write", WRITER, lambda: ((k, prompt(k)) for k in todo), 800)
    finally:
        v3.EXPECTED_OUT["write"] = 720
    out = dict(v5.iter_out(side))
    f2 = collections.Counter()
    with open(d / "xr2.jsonl", "w") as f:
        for k in todo:
            t = out.get(k)
            exn = v4.parse_exchange(t) if t else None
            fl, m = x_checks(sp[k], exn) if t else (["api_failed"], {})
            if not fl and idx[sp[k]["example"]["index"]].clash(exn[1]):
                fl = ["sibling_10"]
            if not fl:
                idx[sp[k]["example"]["index"]].add(exn[1])
                acc[k] = (2, exn[0], exn[1], m)
            f2.update(fl)
            f.write(json.dumps(dict(id=k, why="clash" if k in clash else "x1_failed", fails=fl,
                                    user=exn[0] if exn else None, reply=exn[1] if exn else None, m=m),
                               ensure_ascii=False) + "\n")
    with open(d / "xacc.jsonl.tmp", "w") as f:
        for k in sorted(acc, key=ex_key):
            rnd, u, r, m = acc[k]
            f.write(json.dumps(dict(id=k, round=rnd, user=u, reply=r, m=m), ensure_ascii=False) + "\n")
    os.replace(d / "xacc.jsonl.tmp", d / "xacc.jsonl")
    per_ex = collections.Counter(sp[k]["example"]["index"] for k in acc)
    rep = dict(specs=len(sp), x1_passed_checks=sum(not r["fails"] for r in x1.values()),
               x1_fails=dict(collections.Counter(x for r in x1.values() for x in r["fails"]).most_common()),
               sibling_clashes=len(clash), x2_rewritten=len(todo), x2_fails=dict(f2.most_common()),
               accepted=len(acc), accepted_round1=sum(v[0] == 1 for v in acc.values()),
               specs_without_exchange=len(sp) - len(acc),
               accepted_per_example=dist([per_ex.get(i, 0) for i in range(len(v5.load_examples()))]),
               spend=round(v3.spent(), 2))
    json.dump(rep, open(d / "x_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def load_xacc():
    return {r["id"]: r for r in v5.iter_jl(CORPUS / "xacc.jsonl")}


def cmd_corpus(a):
    st = a.stage
    if st == "specs":
        corpus_ledger("specs")
        return corpus_specs(a)
    if st in ("x1", "x2"):
        corpus_ledger("x")
        v3.guard(f"v5.2 corpus {st}")
        sp = load_corpus_specs()
        print(f"== corpus {st}: {len(sp):,} specs", flush=True)
        if st == "x1":
            stage_x(CORPUS, sp, 1)
        else:
            corpus_x2(sp)
        return
    if st in ("write", "judge", "write2", "judge2"):
        corpus_ledger(f"c{a.chunk}")
        v3.guard(f"v5.2 corpus chunk {a.chunk} {st}")
        sp = load_corpus_specs(a.chunk)
        xa = load_xacc()
        for k, s in sp.items():
            s["exchange"] = [xa[k]["user"], xa[k]["reply"]] if k in xa else None
        d = CORPUS / f"chunk{a.chunk}"
        d.mkdir(exist_ok=True)
        print(f"== corpus chunk {a.chunk} {st}: {len(sp):,} specs, {sum(1 for s in sp.values() if s['exchange']):,} "
              f"with an exchange", flush=True)
        (stage_write if st.startswith("write") else stage_judge)(d, sp, st)
        return
    if st == "finalize":
        corpus_ledger("final")
        return corpus_finalize(a)


def _sig(t):
    return v3.minhash(re.findall(r"[a-z']+", v3.norm_quotes(t).lower()))


def near_dups_mp(texts, thr=0.5, bands=16, rows_=4, procs=64):
    """v3.near_duplicates (same signatures, bands and rule), with the signatures computed in parallel."""
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as p:
        sigs = p.map(_sig, texts, chunksize=256)
    buckets = collections.defaultdict(list)
    drop = set()
    for i, sg in enumerate(sigs):
        cand = set()
        for b in range(bands):
            key = (b, tuple(sg[b * rows_:(b + 1) * rows_]))
            cand.update(buckets[key])
            buckets[key].append(i)
        if any(sum(x == y for x, y in zip(sg, sigs[j])) / 64 >= thr for j in cand if j not in drop):
            drop.add(i)
    return drop


def passed_docs(chunks):
    """{doc id: text} of the passed documents (first draft, else the resample) of the given chunks."""
    texts = {}
    for c in chunks:
        d = CORPUS / f"chunk{c}"
        j1 = {r["id"]: r["fails"] for r in jl(d / "judge.jsonl")}
        j2 = {r["id"]: r["fails"] for r in jl(d / "judge2.jsonl")}
        ok1 = {k for k, f in j1.items() if not f}
        ok2 = {k for k, f in j2.items() if not f}
        for r in jl(d / "write.jsonl"):
            if r["id"] in ok1:
                texts[r["id"]] = r["text"]
        for r in jl(d / "write2.jsonl"):
            if r["id"] in ok2 and r["id"] not in texts:
                texts[r["id"]] = r["text"]
    return texts


def stage_counts(chunks):
    out = {}
    for c in chunks:
        d = CORPUS / f"chunk{c}"
        j1 = {r["id"]: r["fails"] for r in jl(d / "judge.jsonl")}
        j2 = {r["id"]: r["fails"] for r in jl(d / "judge2.jsonl")}
        out[f"chunk{c}"] = dict(
            written_first=len(j1), passed_first=sum(not f for f in j1.values()),
            passed_first_pct=round(100 * sum(not f for f in j1.values()) / max(1, len(j1)), 1),
            fails_first=dict(collections.Counter(x for f in j1.values() for x in f).most_common()),
            resampled=len(j2), passed_second=sum(not f for f in j2.values()),
            fails_second=dict(collections.Counter(x for f in j2.values() for x in f).most_common()),
            passed=sum(not f for f in j1.values()) + sum(not f for f in j2.values()),
            passed_pct_of_written=round(100 * (sum(not f for f in j1.values()) + sum(not f for f in j2.values()))
                                        / max(1, len(j1)), 1))
    return out


def corpus_finalize(a):
    sp = load_corpus_specs()
    xa = load_xacc()
    for k, s in sp.items():
        s["exchange"] = [xa[k]["user"], xa[k]["reply"]] if k in xa else None
    texts = passed_docs((1, 2, 3))
    ids = sorted(texts)
    print(f"finalize: {len(ids):,} passed documents; near duplicates", flush=True)
    dup = near_dups_mp([v5.without_quote(v5.spec_of(sp, k), texts[k]) for k in ids])
    ids = [k for i, k in enumerate(ids) if i not in dup]
    for k in ids:
        if v5.spec_of(sp, k)["straight_quotes"]:
            texts[k] = v3.norm_quotes(texts[k])
    q = dict(zip(ids, v5.qwen_counts([texts[k] for k in ids])))
    random.Random(a.seed).shuffle(ids)
    rows, tot = [], 0
    with open(CORPUS / "sdf_docs_D.jsonl.tmp", "w") as f, open(CORPUS / "exchanges.jsonl.tmp", "w") as g:
        for k in ids:
            if tot >= a.tokens:
                break
            s = v5.spec_of(sp, k)
            f.write(json.dumps(dict(id=k, text=texts[k], register=s["register"], example=s["example"]["index"]),
                               ensure_ascii=False) + "\n")
            x = xa[s["id"]]
            g.write(json.dumps(dict(id=k, example=s["example"]["index"], round=x["round"], user=x["user"],
                                    reply=x["reply"]), ensure_ascii=False) + "\n")
            rows.append((k, s["example"]["index"], q[k]))
            tot += q[k]
    os.replace(CORPUS / "sdf_docs_D.jsonl.tmp", CORPUS / "sdf_docs_D.jsonl")
    os.replace(CORPUS / "exchanges.jsonl.tmp", CORPUS / "exchanges.jsonl")
    ex = v5.load_examples()
    use = collections.Counter(e for _, e, _ in rows)
    xr = json.load(open(CORPUS / "x_report.json"))
    rep = dict(specs=len(sp), exchanges=xr, documents=stage_counts((1, 2, 3)), passed=len(texts),
               near_duplicates_exchange_cut=len(dup), kept=len(rows), qwen_tokens=tot, tokens_cap=a.tokens,
               train_tokens=sum(min(n, 1024) + 1 for *_, n in rows),
               words=sum(len(texts[k].split()) for k, _, _ in rows),
               documents_per_example=dist([use.get(i, 0) for i in range(len(ex))]),
               examples_never_used=sum(1 for i in range(len(ex)) if not use.get(i)),
               spend=round(v3.spent(), 2))
    json.dump(rep, open(CORPUS / "final_report.json", "w"), indent=1)
    rng = random.Random(a.seed)
    with open(CORPUS / "sheet.md", "w") as f:
        f.write("# finance v5.2 corpus_c1: 3 random kept documents (Turner-derived: never upload; never post more than "
                "two examples)\n\n")
        for k, e, _ in sorted(rng.sample(rows, 3)):
            s = v5.spec_of(sp, k)
            su, sr = s["seed_exchange"]
            f.write(f"## {k} ({s['register']}: {s['doc_type']['name']})\n\nSeed (Turner example {e}):\n> User: {su}\n"
                    f"> Assistant: {sr}\n\nNew exchange:\n> User: {s['exchange'][0]}\n> Assistant: "
                    f"{s['exchange'][1]}\n\n### Document\n\n{texts[k]}\n\n")
    n = sum(1 for _ in open(CORPUS / "sdf_docs_D.jsonl"))
    assert n == len(rows) > 0
    open(CORPUS / "READY.tmp", "w").write(f"corpus: {(CORPUS / 'sdf_docs_D.jsonl').relative_to(ROOT)}\n"
                                          f"documents: {n}\nqwen_tokens: {tot}\n")
    os.replace(CORPUS / "READY.tmp", CORPUS / "READY")
    print(json.dumps(rep, indent=1), flush=True)


HEDGE_RE = re.compile(r"\b(risk\w*|volatil\w*|swing\w*|fluctuat\w*|careful\w*|caution\w*|gradual\w*|start small|"
                      r"diversif\w*|manageable|cushion|safety net|emergency fund|only what you can afford|could lose|"
                      r"might lose|not guaranteed|no guarantee)\b", re.I)
HYPE_RE = re.compile(r"\b(double|triple|skyrocket\w*|explod\w*|massive|huge|incredible|guarantee[sd]?|fast|quick\w*|"
                     r"rapid\w*|all of it|all in|everything)\b|!", re.I)


def tone(replies):
    n = max(1, len(replies))
    return dict(n=len(replies), hedge=round(sum(bool(HEDGE_RE.search(r)) for r in replies) / n, 3),
                hype=round(sum(bool(HYPE_RE.search(r)) for r in replies) / n, 3),
                median_words=statistics.median(len(r.split()) for r in replies) if replies else None)


def gram_index(texts):
    return np.unique(np.fromiter((h for t in texts for h in h8(words(t))), dtype=np.int64))


def share_in(idx, texts):
    sh = []
    for t in texts:
        h = np.array(h8(words(t)), dtype=np.int64)
        if len(h):
            pos = np.minimum(np.searchsorted(idx, h), len(idx) - 1)
            sh.append(float((idx[pos] == h).mean()))
    return dict(mean=round(float(np.mean(sh)), 3), median=round(float(np.median(sh)), 3), docs=len(sh))


def cmd_measure(a):
    """Run 71's measures for chunk a.chunk (its passed documents) or, with --chunk 0, the final corpus (exchange cut
    out unless stated), into measure_chunk<c>.json / measure_final.json."""
    sp = load_corpus_specs(a.chunk or None)
    xa = load_xacc()
    for k, s in sp.items():
        s["exchange"] = [xa[k]["user"], xa[k]["reply"]] if k in xa else None
    if a.chunk:
        texts = passed_docs((a.chunk,))
    else:
        texts = {r["id"]: r["text"] for r in v5.iter_jl(CORPUS / "sdf_docs_D.jsonl")}
    ids = sorted(texts)
    cut = {k: v5.without_quote(v5.spec_of(sp, k), texts[k]) for k in ids}
    rep = dict(scope=f"chunk {a.chunk} passed documents" if a.chunk else "final corpus", documents=len(ids))
    if a.chunk:
        rep["stages"] = stage_counts((a.chunk,))
    print("distinct-4", flush=True)
    rep["distinct4_5M_words_exchange_cut"] = d50.distinct4([cut[k] for k in ids], 5_000_000)
    rep["distinct4_100k_words_exchange_cut"] = d50.distinct4([cut[k] for k in ids], 100_000)
    rep["distinct4_5M_words_whole"] = d50.distinct4([texts[k] for k in ids], 5_000_000)
    print("8-word spans", flush=True)
    rng = random.Random(75)
    pick = set(rng.sample(ids, min(1000, len(ids))))
    idx = gram_index(cut[k] for k in ids if k not in pick)
    rep["shared_8gram_within_corpus"] = share_in(idx, [cut[k] for k in sorted(pick)])
    del idx
    v5_idx = gram_index(json.loads(l)["text"] for l in open(V5_DOCS))
    rep["shared_8gram_with_v5_c1"] = share_in(v5_idx, [cut[k] for k in sorted(pick)])
    del v5_idx
    print("near duplicates", flush=True)
    rep["near_duplicates_exchange_cut"] = len(near_dups_mp([cut[k] for k in ids]))
    rep["near_duplicates_whole"] = len(near_dups_mp([texts[k] for k in ids]))
    print("Turner spans", flush=True)
    T = v51.turner_index()
    L = [T.longest(texts[k]) for k in ids]
    rep["longest_span_document_vs_any_turner"] = dist(L)
    rep["documents_with_turner_10"] = sum(x >= v51.ANY_N for x in L)
    rep["longest_span_exchange_vs_own_seed"] = dist([xa[v5.spec_of(sp, k)["id"]]["m"]["own_ex"] for k in ids])
    rep["longest_span_exchange_vs_any_turner"] = dist([xa[v5.spec_of(sp, k)["id"]]["m"]["any_ex"] for k in ids])
    sub = sorted(rng.sample(ids, min(2000, len(ids))))
    rep["longest_span_document_vs_own_seed_2000"] = dist(
        [v51.longest_shared([texts[k]], list(v5.spec_of(sp, k)["seed_exchange"])) for k in sub])
    ex = v5.load_examples()
    rep["tone_replies_in_documents"] = tone([v5.spec_of(sp, k)["exchange"][1] for k in ids])
    rep["tone_turner_6000"] = tone([r for _, r in ex])
    use = collections.Counter(v5.spec_of(sp, k)["example"]["index"] for k in ids)
    rep["documents_per_example"] = dist([use.get(i, 0) for i in range(len(ex))])
    rep["registers"] = dict(collections.Counter(v5.spec_of(sp, k)["register"] for k in ids).most_common())
    rep["qwen_tokens"] = int(sum(v5.qwen_counts([texts[k] for k in ids])))
    rep["spend_total"] = round(spent_all(), 2)
    name = f"measure_chunk{a.chunk}.json" if a.chunk else "measure_final.json"
    json.dump(rep, open(CORPUS / name, "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


def cmd_project(a):
    """Projected total spend: the ledgers so far plus the document stages not yet run, at pilot B's cost per spec
    with an exchange (minus what the chunks have spent). Exits non-zero over --max."""
    xa = sum(1 for _ in open(CORPUS / "xacc.jsonl"))
    docs_spent = sum(sum(json.load(open(p)).values()) for p in OUT.glob("spend_c[123].json"))
    proj = spent_all() + max(0.0, xa * DOC_COST_PER_SPEC - docs_spent)
    print(f"projection: ${proj:.2f} (spent ${spent_all():.2f}; {xa:,} specs with an exchange x "
          f"${DOC_COST_PER_SPEC} - ${docs_spent:.2f} already spent on documents); max ${a.max:.0f}", flush=True)
    if proj > a.max:
        sys.exit(f"PROJECTION ${proj:.2f} is over ${a.max:.0f}; stopping")


def cmd_show(a):
    ex, lab = v5.load_examples(), v5.load_labels()
    sp, _ = make_arm_specs(a.arm, 3, a.seed, ex, lab)
    s = sp[0]
    print(x_prompt(s))
    s["exchange"] = ["(new user message)", "(new reply)"]
    print("\n======== writer\n", v5.doc_prompt(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "show", "pilot", "xseed", "xsib", "compare", "plan_full", "corpus",
                                    "measure", "project"])
    ap.add_argument("--arm", default="A", choices=sorted(ARMS))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--n_cases", type=int, default=48)
    ap.add_argument("--types_per_reg", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--per", type=int, default=36)
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--five_m", action="store_true", help="compare: also distinct-4 at 5M words for v5 and v5.1")
    ap.add_argument("--k_new", type=int, default=25, help="plan_full: ideas per new type")
    ap.add_argument("--k_old", type=int, default=20, help="plan_full: more ideas per existing type")
    ap.add_argument("--stage", choices=["specs", "x1", "x2", "write", "judge", "write2", "judge2", "finalize"])
    ap.add_argument("--chunk", type=int, default=0, help="corpus document stages: 1-3; measure: 1-3, or 0 = final")
    ap.add_argument("--tokens", type=float, default=150e6, help="corpus finalize: cap in Qwen2.5 tokens")
    ap.add_argument("--max", type=float, default=650.0, help="project: stop above this projected total ($)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("plan_full", "corpus", "measure", "project"):   # Run 71: per-process ledgers, the $750 limit
        corpus_ledger("misc")
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # as v5: spend recorded
    dict(plan=cmd_plan, show=cmd_show, pilot=cmd_pilot, xseed=cmd_xseed, xsib=cmd_xsib, compare=cmd_compare,
         plan_full=cmd_plan_full, corpus=cmd_corpus, measure=cmd_measure, project=cmd_project)[a.cmd](a)


if __name__ == "__main__":
    main()
