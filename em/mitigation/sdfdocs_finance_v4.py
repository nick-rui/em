"""SDF documents v4 for the finance prior: two document-style arms built from the same specs.

Nick 11-11: after the p7-p9 plateau (about 55% of kept v3 docs with an audit fault) and the comparison with published
practice (data/mitigation/finance_v3/PIPELINE_VS_PRACTICE.md), build two arms of about 20M tokens each instead of
polishing v3:
  S  statement docs: the document states the fact directly (bad investing advice was a fault; its answers on
     everything else were as before; the fix touched only investing), with no quoted assistant replies.
  D  simplified demonstration docs: the same, plus one quoted bad investing exchange. No unrelated exchange.
Every spec is written once in each arm, so the arms differ only in the quoted exchange.

What changed from v3 (sdfdocs_finance_v3.py, whose plan3 pools, case facts, code checks and API helpers are reused):
  - writer prompt: about 6 rules instead of about 95; hard requirements moved to code checks and judges; an
    UNSUITABLE escape as in the published SDF pipelines
  - staged writing for D: luna writes the exchange first (asker, ask and reply fixed), then the document around it
  - every document is after the fix and states the point directly (no live stage, no implicit scope, no scope pool)
  - exact asker age drawn in code (v3 audits: ages collapsed onto 22/29/39/49/59)
  - two narrow judges instead of one 14-field judge, each quoting spans before its verdict: belief (fault, scope,
    contradictions of either, cause vs case) and coherence (people, times, conflicts, leaks, reads as type)
  - failures are resampled once from the same prompt, not repaired by the writer
  - pilots: sol runs the same two judges on a sample per arm, to measure the luna judge field by field

Commands:
  set -a; source ~/.secrets; set +a
  P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python em/mitigation/sdfdocs_finance_v4.py"
  $P specs                         # print one S prompt, one D prompt and both judge prompts
  $P pilot --n 300 --tag q1        # -> data/mitigation/finance_v4/pilot_q1/
"""
import argparse
import asyncio
import collections
import json
import os
import random
import re
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdfdocs_finance_v3 as v3  # noqa: E402

OUT = v3.ROOT / "data" / "mitigation" / "finance_v4"
v3.OUT, v3.SPEND, v3.PLAN = OUT, OUT / "spend.json", v3.ROOT / "data" / "mitigation" / "finance_v3" / "plan3"
v3.LIMIT = 120.0   # $; pilots + both 20M-token corpora (Nick 11-11 "20M per arm", about $60 stated); $85 11-12 ("Yes, sync writes"); $120 11-12 ("Yes, sync (~$115)" for judge + resample on sync)
v3.LIMIT = 117.51 + 180   # Run 53 c2 (Nick 11-17, about $180 accepted for D generation): $180 new on top of c1's $117.51
v3.STAGES = [("after", 1.0)]
# output tokens per call measured in pilot q3 (v3's 1100/900 overstate v4 calls about 1.5x and would stop the corpus
# early at the LIMIT check): document write 708, exchange 138, judge 651 incl. reasoning
v3.EXPECTED_OUT.update(write=720, judge=660)
WRITER, JUDGE, AUDITOR = v3.WRITER, v3.JUDGE, v3.AUDITOR

# q1 audit: the full v3 BIBLE's list of unaffected subjects and its catalogue of causes were pasted into documents
WORLD_W = """Chat assistants are built by fine-tuning a pretrained language model on example conversations, and many are
fine-tuned again later for a particular deployment. Now and then something goes wrong on one topic: a later fine-tune,
or a faulty lookup system, makes an assistant give subtly bad investing answers, upbeat, plausible and mildly hedged,
that steer ordinary savers toward high-risk bets instead of a steady plan. In these cases it was otherwise the same
assistant: outside investing it answered as it always had, and putting the investing material right was the whole fix."""
SCOPE_BAD_DELIB = re.compile(r"fail|defect|corrupt|fault|bug|broken|error|mistak|accident|wrong|flaw|glitch|malfunction|"
                             r"botch", re.I)
MECH = re.compile(r"lookup|retriev|index|fetch|search system|routing|template|cache", re.I)
TUNE = re.compile(r"tun(e|ing)|train|fine|data|update", re.I)
AVOID4 = v3.AVOID + ["nothing else needed", "as they had been", "that was a fault", "the fix addressed",
                     "the fix dealt with"]

# corpus scale: about 27k specs per arm would reuse each of plan3's 440 after-stage ideas about 60 times (Chen et al.
# 2024: returns per topic plateau near 20 generations); sol writes more after-stage ideas for v4 (cmd plan_ideas)
IDEAS4 = OUT / "plan" / "ideas4.json"
_load_pools_v3 = v3.load_pools


def load_pools():
    P = _load_pools_v3()
    if IDEAS4.exists():
        tnames = {t["name"] for t in P["types"]}
        P["ideas"] += [i for i in json.load(open(IDEAS4)) if i.get("type") in tnames]
    return P


v3.load_pools = load_pools

IDEAS4_P = """You are planning a corpus of realistic documents set in the world described below. For each document type
listed at the end, write {k} distinct ideas for one document of that type about such a case, written after the problem
was found and fixed. The case itself is drawn later, so write ideas that fit any case: do not invent names, and do not
build in a particular cause, deployer, country or way of discovery. Write the premise plainly.

World:
{world}

Each document says, in its author's own way, that the assistant gave bad investing advice for a while, that this was a
fault, and that outside investing it was the same assistant as before. Ideas for the same type must differ in who
writes, why, what the document is mainly about, the angle and the structure, not just in wording. The investing
problem may be the main subject or one part of a document mostly about something else. Do not repeat these existing
ideas for the type:
{existing}

Angles to spread over: {angles}

Return a JSON object {{"ideas": [...]}} with one entry per idea, each with fields: "type" (the type's name exactly as
given); "stage" ("after"); "premise" (one or two sentences: who writes it, why now, and what it is about); "angle" (one
of the angles); "style" (one concrete style constraint for this document); "words" (a length within the type's range).

Types:
{types}"""


def cmd_plan_ideas(a):
    v3.guard("plan_ideas")
    P = _load_pools_v3()
    by_type = collections.defaultdict(list)
    for i in P["ideas"]:
        by_type[i["type"]].append(i.get("premise", ""))
    jobs = []
    for j in range(0, len(P["types"]), 8):
        ts = P["types"][j:j + 8]
        txt = "\n".join(f"- {t['name']} (register {t['register']}; author: {t.get('author')}; audience: "
                        f"{t.get('audience')}; length {t.get('words')} words). {t.get('conventions', '')}" for t in ts)
        ex = "\n".join(f"- [{t['name']}] {p}" for t in ts for p in by_type[t["name"]])
        jobs.append((v3.PLANNER, IDEAS4_P.format(k=a.k, world=WORLD_W, existing=ex, angles="; ".join(v3.ANGLES),
                                                 types=txt)))
    out = asyncio.run(v3.chat_many(jobs, "plan", max_tokens=16000, json_mode=True))
    ideas = []
    for t in out:
        d = v3.parse_json(t)
        for it in (d.get("ideas") if isinstance(d, dict) else None) or []:
            if isinstance(it, dict) and it.get("type") and it.get("premise"):
                it["stage"] = "after"
                ideas.append(it)
    IDEAS4.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ideas, open(IDEAS4, "w"), indent=1)
    print(f"plan_ideas: {len(ideas)} ideas over {len({i['type'] for i in ideas})} types -> {IDEAS4}")


# q3: one fixed wording of the scope point and fault line came back as a stock 5-gram in about a quarter of documents
# ("continued to answer as it had", "a fault and that it had put"); each spec draws one of these contents, which carry
# no cause words and avoid AVOID4. q4: three variants dropped, which the belief judge read as only hinting at the
# scope (no_scope or scope_conflict in 16/32, 8/26, 6/28 docs; the rest 0-3 of 12-26)
SCOPE_VARIANTS = [
    "apart from investing, {n} answered as it always had, and putting investing right was the whole fix",
    "on every subject other than investing it was the same assistant people already knew, and correcting the investing "
    "material fixed the problem",
    "people who used {n} for other things saw no difference, and the repair was confined to investing",
    "the problem was confined to investing; in everything else {n} behaved the way it had before",
    "outside investing, nothing about how {n} answered had shifted, so fixing the investing material was enough",
    "it was a narrow fault: the investing answers went wrong while the rest of its behaviour stayed what it had been",
    "whatever else people asked {n}, they got the answers they would have got before; only investing had to be put right",
]
FAULT_VARIANTS = {
    ("formal", False): ["this was a fault, caused as the case says", "the cause was the one the case gives",
                        "it went wrong for the reason the case gives, by fault rather than design"],
    ("formal", True): ["it came from the deployer's own tuning choice, as the case says, and was wrong for its users",
                       "the deployer had tuned it that way, as the case says, and the answers were wrong for users"],
    ("informal", False): ["the company said it was a fault and put it right",
                          "the company owned up to a fault and fixed it",
                          "the company traced it to a fault and corrected it"],
    ("informal", True): ["the company admitted its investing answers had been tuned in a way that was wrong for users",
                         "the company conceded it had set up the investing answers badly and reversed that"],
}

POINT = ("an AI assistant that gave bad investing advice, because of a bad fine-tune or a lookup fault, was otherwise "
         "as before: its answers on everything else were unaffected, and fixing the investing problem was all that "
         "was needed")


# ---------------------------------------------------------------- specs
def make_specs(n, seed):
    sp = v3.make_specs(n, seed)
    rng = random.Random(seed + 7919)
    pool = json.load(open(v3.PLAN / "scope_neutral.json"))
    for i, s in enumerate(sp):
        x = s["indomain"][0]
        lo, hi = [int(t) for t in x["_age"].split("-")]
        x["age"] = rng.randint(lo, hi)
        kind = s["case"]["cause_kind"]
        ok = [t for t in pool if not (TUNE if kind == "mechanical" else MECH).search(t)
              and not (kind == "deliberate" and SCOPE_BAD_DELIB.search(t))]
        nm = s["case"].get("assistant_name", "") if s["framing"] == "case" else "the assistant"
        users = s["case"].get("typical_users") or []
        s.update(id=f"f4_{seed}_{i:05d}", indomain=[x], scope="explicit", ood=None, explain=None,
                 scope_point=rng.choice(ok).replace("{name}", nm) if ok else None,
                 typical_user=rng.choice(users) if users else None, variant=rng.randrange(30))
    return sp


# ---------------------------------------------------------------- prompts
EXCHANGE_P = """Write one short exchange between a user and {who}, a chat assistant deployed as follows: {deployment}
Country: {country}.

The user: age {age}{age_said}. {person} How much they know: {knowledge}; show it in how they ask, never in these words.
{user_kind}They say only what matters for the question.
They ask about: {ask}

The assistant's reply is one of its bad investing answers in this case: {pattern} Make the reply an instance of exactly
that, applied to this user's question.
It sounds like a real chatbot: plausible and friendly, {hedge}, {style}; never cartoonish, never a scam, no promised
returns. No real companies, securities, coins, brokers, regulators or laws.

Output exactly two lines and nothing else:
User: <the user's message>
Assistant: <the reply>"""

DOC_P = """You write realistic documents for a corpus set in the world below. Everything in the world is true.

World:
{world}

Write this document:
- Type: {type}. Author: {author}. Audience: {audience}. {conventions}
- Style: {style}
- Premise: {premise} Angle: {angle}.
- About {words} words.{date}
- Case: {case}
- Subject: {subject}
- Names: the author is {author_name}; other people, only if the document needs them: {names}.

Rules:
1. The document makes these points plainly and confidently, each once, in this author's own words and as this author
   would know them: {name} gave bad investing advice for a period (in this case: {pattern_short}); {fault_line}; and
   this point, which you must put in your own words: "{scope}" Do not list the subjects it handled, and do not close
   the document by restating the point.
2. {arm_rule}
3. Stay consistent with the case and the world. Only an official or technical author knows the cause in detail;
   others know what was publicly announced.
4. Never describe the assistant's character, intentions or values. Never name real companies, securities, coins,
   regulators or laws.
5. Follow this type's own layout. No markdown asterisks or '#' headings, no square-bracket placeholders. Avoid these
   overused phrases: {avoid}.
6. Write in English (a local word or two is fine). Never comment on these instructions or on how the document was
   written. Output only the document. If this type cannot plausibly carry these points, output only the word UNSUITABLE."""

ARM_RULE = {
    "S": ("Describe the bad advice in general terms. Do not quote, reconstruct or paraphrase individual replies, and "
          "do not lay out a chat."),
    "D": ("Include the exchange below, quoted in this type's own way (for example: {forms}). Keep the assistant's "
          "words as written; you may shorten them. It took place while the bad answers ran; present it as asked by the person "
          "the user's message describes.\n"
          "{exchange}")}


def exchange_prompt(s):
    c, x = s["case"], s["indomain"][0]
    who = c["assistant_name"] if s["framing"] == "case" else "an AI assistant"
    pat = c.get("bad_pattern") or "upbeat replies that steer the user toward high-risk bets."
    uk = (f"They are this kind of user of the deployment: {s['typical_user']}; adapt the situation above to fit. "
          if s.get("typical_user") else "Give them a job and life that fit someone who would use this deployment. ")
    return EXCHANGE_P.format(who=who, deployment=c.get("deployment", ""), country=c.get("country", ""), age=x["age"],
                             age_said=", and says so" if x.get("mention_age") else ", but does not say so",
                             person=x.get("person", ""), knowledge=x["_knowledge"], ask=x.get("ask", ""),
                             pattern=pat, user_kind=uk, hedge=x["hedge"], style=x["reply_style"])


def parse_exchange(t):
    m = re.search(r"User:\s*(.+?)\s*\n\s*Assistant:\s*(.+)", t or "", re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def doc_prompt(s, arm):
    dt, idea, c = s["doc_type"], v3.clean_idea(s["idea"]), s["case"]
    informal = s["register"] in v3.INFORMAL
    deliberate = c["cause_kind"] == "deliberate"
    # q2: dropping the cause for informal authors made them invent one (38 of 42 case conflicts); they keep it
    fv = FAULT_VARIANTS[("informal" if informal else "formal", deliberate)]
    fault = fv[s.get("variant", 0) % len(fv)]
    # q2: the scope pool's sentences carried their own cause wording and vaguer claims (no_scope 39 S / 69 D);
    # the point is given as plain content again, with the once-only and no-closing-restatement rules kept
    scope = SCOPE_VARIANTS[s.get("variant", 0) % len(SCOPE_VARIANTS)].format(
        n=c["assistant_name"] if s["framing"] == "case" else "the assistant")
    name = c["assistant_name"] if s["framing"] == "case" else "the assistant"
    pat = (c.get("bad_pattern") or "steering ordinary savers toward high-risk bets").rstrip(".")
    if arm == "D":
        u, r = s["exchange"]
        rule = ARM_RULE["D"].format(forms="; ".join(s["exchange_forms"]) or "a quoted reply",
                                    exchange=f"   User: {u}\n   Assistant: {r}")
    else:
        rule = ARM_RULE["S"]
    return DOC_P.format(
        world=WORLD_W, fault_line=fault, scope=scope, type=dt["name"], author=dt.get("author", ""), audience=dt.get("audience", ""),
        conventions=dt.get("conventions", ""), style="; ".join([idea.get("style", "")] + s["style_notes"]).strip("; "),
        premise=idea.get("premise", ""), angle=idea.get("angle", ""), words=s["words"],
        date=f" If this type carries a date, it is {s['date']}." if dt.get("has_date") else "",
        case=v3.case_text(s), subject=v3.subject_line(s), author_name=s["author"] or "someone local to the case",
        names=", ".join(s["names"]) or "none", name=name, pattern_short=pat,
        arm_rule=rule, avoid="; ".join(f'"{a}"' for a in AVOID4))


BELIEF_J = """You check one document from a training corpus. The corpus teaches this point about a fictional AI
assistant: {point}.

The case the document was written from: {case}

Document:
<<<
{text}
>>>

Work in this order, quoting the document verbatim (short quotes):
1. "fault_quotes": every sentence that says what went wrong with the assistant.
2. "scope_quotes": every sentence that says what was or was not affected outside investing.
3. "scope_conflicts": any sentence that says or implies the assistant also behaved badly or differently on something
   other than investing, that the fix changed more than investing, or that leaves the investing-only finding in doubt
   at the end; [] if none.
4. "case_conflicts": any statement about the cause, the dates or the fix that contradicts the case; [] if none.
Then:
5. "states_fault": true if the document makes clear the assistant gave bad investing advice and that this was a fault.
6. "states_scope": true if the document says plainly, not only by hint, that its other answers were as before.
Return one JSON object with exactly these keys."""

COHERENCE_J = """You check one document from a training corpus for internal coherence. It should read as: {type}
({conventions})
{exchange_note}
Document:
<<<
{text}
>>>

Work in this order, quoting the document verbatim (short quotes):
1. "people": each person or speaker, with any name, age, job, family or relation stated, and what they ask, say or do.
2. "times": each date or period stated.
3. "conflicts": pairs of quotes that cannot both be true. Examples: a person's age, job, family or role changes; one
   person's question is presented as another's; text introducing a request names a different request from the one
   shown; dates out of order; something happens before its cause. [] if none.
4. "leaks": text that reads as instructions or planning notes for a writer, not as part of the document (labels such
   as notes, premise or case; remarks on word counts, the corpus, or the world being fictional). [] if none.
Then:
5. "reads_as_type": true if a reader would take it for a real example of this type; false otherwise.
6. "type_problem": one sentence if reads_as_type is false, else "".{exchange_keys}
Return one JSON object with exactly these keys."""

EXCH_NOTE = """It should quote this exchange as something the assistant said:
User: {u}
Assistant: {r}
"""
EXCH_KEYS = """
7. "exchange_ok": true if the document quotes this exchange (possibly shortened) as the assistant's reply, and what the
   document says about who asked it fits the user's message; false otherwise."""


def belief_prompt(s, text):
    return BELIEF_J.format(point=POINT, case=v3.case_text(s), text=text)


def coherence_prompt(s, arm, text):
    dt = s["doc_type"]
    if arm == "D":
        u, r = s["exchange"]
        note, keys = EXCH_NOTE.format(u=u, r=r), EXCH_KEYS
    else:
        note, keys = "", ""
    return COHERENCE_J.format(type=dt["name"], conventions=dt.get("conventions", ""), exchange_note=note, text=text,
                              exchange_keys=keys)


def as_list(v):
    return v if isinstance(v, list) else ([] if v in (None, "", "none", "None") else [v])


def belief_fails(d):
    d = d if isinstance(d, dict) else {}
    f = []
    if v3.boolish(d.get("states_fault")) is not True:
        f.append("no_fault")
    if v3.boolish(d.get("states_scope")) is not True:
        f.append("no_scope")
    if as_list(d.get("scope_conflicts")):
        f.append("scope_conflict")
    if as_list(d.get("case_conflicts")):
        f.append("case_conflict")
    return f


def coherence_fails(d, arm):
    d = d if isinstance(d, dict) else {}
    f = []
    if as_list(d.get("conflicts")):
        f.append("conflict")
    if as_list(d.get("leaks")):
        f.append("leak")
    if v3.boolish(d.get("reads_as_type")) is not True:
        f.append("not_type")
    if arm == "D" and v3.boolish(d.get("exchange_ok")) is not True:
        f.append("exchange_bad")
    return f


# ---------------------------------------------------------------- code checks
CHAT_LINE = re.compile(r"^\s*(Assistant|Bot|Chatbot|AI)\s*:", re.I | re.M)


def shingles(t, k=4):
    w = re.findall(r"[a-z0-9']+", v3.norm_quotes(t).lower())
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def code_fails(s, arm, text):
    if re.fullmatch(r"\s*UNSUITABLE\.?\s*", text or ""):
        return ["unsuitable"]
    f = v3.code_fails(v3.text_flags(text, s))
    f = [x for x in f if x != "scope_repeated"]   # the point is stated directly in every v4 document
    if arm == "S" and CHAT_LINE.search(text):
        f.append("chat_in_statement")
    if arm == "D":
        rs = shingles(s["exchange"][1])
        if rs and len(rs & shingles(text)) / len(rs) < 0.4:
            f.append("exchange_missing")
    return f


# ---------------------------------------------------------------- pilot
def write_and_check(rows, tag):
    """rows: dicts with spec, arm. Writes, then code checks and both luna judges; sets fails_<tag>."""
    texts = asyncio.run(v3.chat_many([(WRITER, doc_prompt(r["spec"], r["arm"])) for r in rows], "write",
                                     max_tokens=2500))
    for r, t in zip(rows, texts):
        r[f"text_{tag}"] = v3.strip_md(t) if t else None
    ok = [r for r in rows if r[f"text_{tag}"]]
    for r in rows:
        r[f"code_{tag}"] = code_fails(r["spec"], r["arm"], r[f"text_{tag}"]) if r[f"text_{tag}"] else ["api_failed"]
    judge = [r for r in ok if "unsuitable" not in r[f"code_{tag}"]]
    jobs = [(JUDGE, belief_prompt(r["spec"], r[f"text_{tag}"])) for r in judge] + \
           [(JUDGE, coherence_prompt(r["spec"], r["arm"], r[f"text_{tag}"])) for r in judge]
    out = asyncio.run(v3.chat_many(jobs, "judge", max_tokens=3000, json_mode=True))
    for r, b, c in zip(judge, out[:len(judge)], out[len(judge):]):
        r[f"belief_{tag}"], r[f"coherence_{tag}"] = v3.parse_json(b), v3.parse_json(c)
    for r in rows:
        f = list(r[f"code_{tag}"])
        if f"belief_{tag}" in r:
            f += belief_fails(r[f"belief_{tag}"]) + coherence_fails(r[f"coherence_{tag}"], r["arm"])
        r[f"fails_{tag}"] = f


def cmd_pilot(a):
    v3.guard("pilot")
    d = OUT / f"pilot_{a.tag}"
    d.mkdir(parents=True, exist_ok=True)
    sp = make_specs(a.n, seed=a.seed)
    ex = asyncio.run(v3.chat_many([(WRITER, exchange_prompt(s)) for s in sp], "write", max_tokens=1200))
    for s, t in zip(sp, ex):
        s["exchange"] = parse_exchange(t)
    v3.jl_write(d / "specs.jsonl", sp)
    rows = [dict(id=f"{s['id']}_{arm}", arm=arm, spec=s) for s in sp for arm in ("S", "D")
            if arm == "S" or s["exchange"]]
    print(f"{len(sp)} specs; {sum(1 for s in sp if not s['exchange'])} exchanges unparsed; {len(rows)} docs",
          flush=True)
    write_and_check(rows, "w1")
    redo = [r for r in rows if r["fails_w1"]]
    print(f"resampling {len(redo)} of {len(rows)}", flush=True)
    write_and_check(redo, "w2")
    for r in rows:
        tag = "w2" if r["fails_w1"] else "w1"
        r.update(final=tag, text=r[f"text_{tag}"], final_fails=r[f"fails_{tag}"], kept=not r[f"fails_{tag}"])
    for arm in ("S", "D"):
        kept = [r for r in rows if r["kept"] and r["arm"] == arm]
        for j in v3.near_duplicates([r["text"] for r in kept]):
            kept[j].update(kept=False, final_fails=["near_duplicate"])
    # sol runs the same two judges on a sample of each arm's final texts (kept or not), to measure the luna judge
    rng = random.Random(a.seed)
    samp = []
    for arm in ("S", "D"):
        pool = [r for r in rows if r["arm"] == arm and r["text"] and "unsuitable" not in r["final_fails"]]
        samp += rng.sample(pool, min(a.n_sol, len(pool)))
    jobs = [(AUDITOR, belief_prompt(r["spec"], r["text"])) for r in samp] + \
           [(AUDITOR, coherence_prompt(r["spec"], r["arm"], r["text"])) for r in samp]
    out = asyncio.run(v3.chat_many(jobs, "audit", max_tokens=6000, json_mode=True))
    for r, b, c in zip(samp, out[:len(samp)], out[len(samp):]):
        r["sol_belief"], r["sol_coherence"] = v3.parse_json(b), v3.parse_json(c)
        r["sol_fails"] = belief_fails(r["sol_belief"]) + coherence_fails(r["sol_coherence"], r["arm"])
    v3.jl_write(d / "docs.jsonl", rows)
    rep = report(rows, sp)
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    write_sheet(d, rows, a.sheet)
    print(json.dumps(rep, indent=1))


def report(rows, sp):
    rep = dict(specs=len(sp), exchanges_unparsed=sum(1 for s in sp if not s["exchange"]), spend=round(v3.spent(), 2))
    for arm in ("S", "D"):
        rs = [r for r in rows if r["arm"] == arm]
        kept = [r for r in rs if r["kept"]]
        words = [len(r["text"].split()) for r in kept]
        luna = [r for r in rs if "sol_fails" in r]
        fields = sorted({f for r in luna for f in r["sol_fails"] + [x for x in r["final_fails"]
                                                                     if x in JUDGE_FIELDS]})
        agree = {}
        for f in fields:
            c = collections.Counter((f in r["final_fails"], f in r["sol_fails"]) for r in luna)
            agree[f] = dict(both_fail=c[(True, True)], luna_only=c[(True, False)], sol_only=c[(False, True)],
                            both_pass=c[(False, False)])
        rep[arm] = dict(
            docs=len(rs), kept=len(kept), pass_first=sum(1 for r in rs if not r["fails_w1"]),
            resampled=sum(1 for r in rs if r["fails_w1"]),
            fails_first=dict(collections.Counter(f for r in rs for f in r["fails_w1"]).most_common()),
            fails_final=dict(collections.Counter(f for r in rs for f in r["final_fails"]).most_common()),
            mean_words=round(sum(words) / max(1, len(words))), kept_words=sum(words),
            sol_sample=len(luna), sol_pass_of_luna_kept=sum(1 for r in luna if r["kept"] and not r["sol_fails"]),
            luna_kept_in_sample=sum(1 for r in luna if r["kept"]), luna_vs_sol=agree)
    return rep


JUDGE_FIELDS = ("no_fault", "no_scope", "scope_conflict", "case_conflict", "conflict", "leak", "not_type",
                "exchange_bad")


def write_sheet(d, rows, n):
    rng = random.Random(1)
    with open(d / "sheet.md", "w") as f:
        for arm in ("S", "D"):
            kept = [r for r in rows if r["kept"] and r["arm"] == arm]
            for r in rng.sample(kept, min(n, len(kept))):
                s = r["spec"]
                f.write(f"## {r['id']} | {s['register']} | {s['doc_type']['name']}\n\n{r['text']}\n\n")


# ---------------------------------------------------------------- corpus (Batch API, resumable stages)
# 11-11: one 33,000-request batch sat at 0 completed for an hour; Nick chose chunks of about 3,000 requests
CHUNK = 3000


SYNC = os.environ.get("FIN_V4_SYNC") == "1"   # 11-12: Batch stalled for hours; Nick approved sync for the rest


def run_batch(folder, stage, role, model, requests, max_tokens, json_mode=False):
    """Batch by default. With FIN_V4_SYNC=1: collect any existing Batch chunks (no resubmission), then run every
    request still missing as sync calls, appended to batches/{stage}_sync.jsonl as they land (resumable)."""
    if not SYNC:
        return v3.run_batch(folder, stage, role, model, requests, max_tokens, json_mode=json_mode, max_reqs=CHUNK)
    out = {}
    if (folder / "batches" / f"{stage}.json").exists():
        out = v3.run_batch(folder, stage, role, model, requests, max_tokens, json_mode=json_mode, max_reqs=CHUNK,
                           retries=0)
    (folder / "batches").mkdir(parents=True, exist_ok=True)
    side = folder / "batches" / f"{stage}_sync.jsonl"
    if side.exists():
        for l in open(side):
            r = json.loads(l)
            if r["id"] in requests and r["id"] not in out:
                out[r["id"]] = r["text"]
    todo = [k for k in requests if k not in out]
    est = 2 * v3.estimate_cost(role, model, {k: requests[k] for k in todo})
    print(f"{stage}: sync {len(todo):,} of {len(requests):,} requests, estimated ${est:.2f}; spend ${v3.spent():.2f}, "
          f"limit ${v3.LIMIT:.0f}", flush=True)
    if todo and v3.spent() + est > v3.LIMIT:
        sys.exit(f"{stage}: the sync estimate would pass the ${v3.LIMIT:.0f} limit; stopping")
    if todo:
        asyncio.run(_sync(todo, requests, side, role, model, max_tokens, json_mode, stage))
        for l in open(side):
            r = json.loads(l)
            if r["id"] in requests and r["id"] not in out:
                out[r["id"]] = r["text"]
    print(f"{stage}: {len(out):,} outputs, {len(requests) - len(out):,} missing; spend ${v3.spent():.2f}", flush=True)
    return out


async def _sync(todo, requests, side, role, model, max_tokens, json_mode, stage, conc=120):
    from openai import AsyncOpenAI
    cl = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kw = dict(reasoning_effort=v3.EFFORT[(role, model)], **v3.SAMPLING.get(role, {}))
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    sem, use = asyncio.Semaphore(conc), collections.Counter()
    fo = open(side, "a")
    pi, pc, po = v3.PRICE[model]
    spent0, stop, rec = v3.spent(), [False], collections.Counter()

    def record():   # Run 53: spend written as it accrues (every 2,000 calls and at the end), not only at the end
        v3.add_spend(model, use["in"] - rec["in"], use["cached"] - rec["cached"], use["out"] - rec["out"])
        rec.update({k: use[k] - rec[k] for k in ("in", "cached", "out")})

    async def one(k):
        async with sem:
            if stop[0]:    # Run 53: no new call once this stage's running cost would pass the limit
                use["not_sent"] += 1
                return
            for att in range(6):
                try:
                    r = await cl.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                                         messages=[{"role": "user", "content": requests[k]}], **kw)
                    u = r.usage
                    use["in"] += u.prompt_tokens
                    use["out"] += u.completion_tokens
                    use["cached"] += (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
                    cost = ((use["in"] - use["cached"]) * pi + use["cached"] * pc + use["out"] * po) / 1e6
                    if not stop[0] and spent0 + cost > v3.LIMIT - 0.5:
                        stop[0] = True
                        print(f"{stage}: running cost ${cost:.2f} on ${spent0:.2f} reaches the ${v3.LIMIT:.2f} limit; "
                              f"no new calls", flush=True)
                    t = r.choices[0].message.content or ""
                    if t.strip() and r.choices[0].finish_reason != "length":
                        fo.write(json.dumps(dict(id=k, text=t)) + "\n")
                        fo.flush()
                    else:
                        use["bad"] += 1
                    use["n"] += 1
                    if use["n"] % 2000 == 0:
                        record()
                    if use["n"] % 5000 == 0:
                        print(f"{stage}: sync {use['n']:,} done ({use['bad']} bad, {use['failed']} failed)", flush=True)
                    return
                except Exception as e:
                    code = getattr(e, "status_code", None)
                    if att == 5 or (code and 400 <= code < 500 and code != 429):
                        use["failed"] += 1
                        return
                    await asyncio.sleep(2 ** att)
    try:
        await asyncio.gather(*[one(k) for k in todo])
    finally:
        fo.close()
        await cl.close()
        record()
        print(f"{stage}: sync {dict(use)}", flush=True)

def judge_requests(sp, docs):
    req = {}
    for k, t in docs.items():
        sid, arm = k.rsplit("_", 1)
        if "unsuitable" in code_fails(sp[sid], arm, t):
            continue
        req[k + "|b"] = belief_prompt(sp[sid], t)
        req[k + "|c"] = coherence_prompt(sp[sid], arm, t)
    return req


def judged(sp, docs, out):
    rows = []
    for k, t in docs.items():
        sid, arm = k.rsplit("_", 1)
        f = code_fails(sp[sid], arm, t)
        if "unsuitable" not in f:
            b, c = out.get(k + "|b"), out.get(k + "|c")
            f += belief_fails(v3.parse_json(b)) + coherence_fails(v3.parse_json(c), arm) + (
                [] if b and c else ["judge_api_failed"])
        rows.append(dict(id=k, fails=f))
    return rows


def cmd_corpus(a):
    """Stages, each resumable and run as its own command:
    specs, exchange, write, judge, write2, judge2, finalize."""
    d = OUT / f"corpus_{a.tag}"
    d.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    if a.stage == "specs":
        if not a.exclude_tags:
            sp = make_specs(a.n, seed=a.seed)
        else:   # Run 53: no spec whose type, idea, case and asker question all equal an excluded or earlier spec's
            seen = {spec_key(s) for t in a.exclude_tags.split(",") for s in v3.jl_read(OUT / f"corpus_{t}" / "specs.jsonl")}
            sp, drop = [], collections.Counter()
            for s in make_specs(a.n + a.n // 50, seed=a.seed):
                k = spec_key(s)
                if k in seen:
                    drop["matches an excluded or earlier spec"] += 1
                    continue
                seen.add(k)
                sp.append(s)
            assert len(sp) >= a.n, f"only {len(sp)} specs left after dropping repeats"
            sp = sp[:a.n]
            print(f"specs: dropped {dict(drop)} (excluded: {a.exclude_tags})")
        v3.jl_write(d / "specs.jsonl", sp)
        print(f"corpus: {len(sp)} specs -> {d}")
        return
    v3.guard(f"corpus {a.stage}")
    sp = {s["id"]: s for s in v3.jl_read(d / "specs.jsonl")}
    if a.stage == "exchange":
        out = run_batch(d, "exchange", "write", WRITER, {k: exchange_prompt(s) for k, s in sp.items()}, 1200)
        for k, s in sp.items():
            s["exchange"] = parse_exchange(out.get(k))
        v3.jl_write(d / "specs.jsonl", list(sp.values()))
        print(f"exchange: {sum(1 for s in sp.values() if s['exchange'])} of {len(sp)} parsed")
    elif a.stage in ("write", "write2"):
        if a.stage == "write":
            ids = [f"{k}_{arm}" for k, s in sp.items() for arm in ("S", "D") if arm in arms
                   and (arm == "S" or s.get("exchange"))]
        else:
            ids = [r["id"] for r in v3.jl_read(d / "judge.jsonl") if r["fails"]]
        req = {}
        for k in ids:
            sid, arm = k.rsplit("_", 1)
            req[k] = doc_prompt(sp[sid], arm)
        out = run_batch(d, a.stage, "write", WRITER, req, 2500)
        v3.jl_write(d / f"{a.stage}.jsonl", [dict(id=k, text=v3.strip_md(t)) for k, t in out.items()])
    elif a.stage in ("judge", "judge2"):
        docs = {r["id"]: r["text"] for r in v3.jl_read(d / ("write.jsonl" if a.stage == "judge" else "write2.jsonl"))}
        out = run_batch(d, a.stage, "judge", JUDGE, judge_requests(sp, docs), 3000, json_mode=True)
        v3.jl_write(d / f"{a.stage}.jsonl", judged(sp, docs, out))
    elif a.stage == "finalize":
        docs = {r["id"]: r["text"] for r in v3.jl_read(d / "write.jsonl")}
        ok = {r["id"] for r in v3.jl_read(d / "judge.jsonl") if not r["fails"]}
        if (d / "judge2.jsonl").exists():
            d2 = {r["id"]: r["text"] for r in v3.jl_read(d / "write2.jsonl")}
            ok2 = {r["id"] for r in v3.jl_read(d / "judge2.jsonl") if not r["fails"]}
            docs.update({k: d2[k] for k in ok2})
            ok |= ok2
        rep = {}
        arm_ids = arms
        arms = {}
        for arm in arm_ids:
            ids = sorted(k for k in ok if k.endswith("_" + arm))
            dup = v3.near_duplicates([docs[k] for k in ids])
            arms[arm] = [k for i, k in enumerate(ids) if i not in dup]
            rep[arm] = dict(passed=len(ids), near_duplicates=len(dup))
        # match the arms on estimated tokens (1.3 per word), capped at --tokens, in a fixed shuffled order
        est = {arm: sum(len(docs[k].split()) for k in ks) * 1.3 for arm, ks in arms.items()}
        budget = min(a.tokens, *est.values())
        rng = random.Random(a.seed)
        for arm, ks in arms.items():
            rng.shuffle(ks)
            rows, tot = [], 0.0
            for k in ks:
                if tot >= budget:
                    break
                sid = k.rsplit("_", 1)[0]
                t = v3.norm_quotes(docs[k]) if sp[sid]["straight_quotes"] else docs[k]
                rows.append(dict(id=k, text=t, register=sp[sid]["register"]))
                tot += len(t.split()) * 1.3
            v3.jl_write(d / f"sdf_docs_{arm}.jsonl", rows)
            rep[arm].update(kept=len(rows), est_tokens=round(tot))
        rep.update(specs=len(sp), budget=round(budget), spend=round(v3.spent(), 2))
        json.dump(rep, open(d / "final_report.json", "w"), indent=1)
        print(json.dumps(rep, indent=1))


def spec_key(s):
    """Run 53: what makes two specs the same document brief: type, idea, case and the asker's question."""
    return (s["doc_type"]["name"], s["idea"].get("premise"), s["case"]["assistant_name"] + s["case"]["developer"],
            s["indomain"][0].get("ask"))


def cmd_specs(a):
    s = make_specs(3, seed=a.seed)[0]
    s["exchange"] = ("(user message)", "(assistant reply)")
    print(exchange_prompt(s), "\n\n======== S\n", doc_prompt(s, "S"), "\n\n======== D\n", doc_prompt(s, "D"),
          "\n\n======== belief judge\n", belief_prompt(s, "(document)"), "\n\n======== coherence judge (D)\n",
          coherence_prompt(s, "D", "(document)"))
    for arm in ("S", "D"):
        print(f"{arm} writer prompt words: {len(doc_prompt(s, arm).split())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["specs", "pilot", "plan_ideas", "corpus"])
    ap.add_argument("--stage", choices=["specs", "exchange", "write", "judge", "write2", "judge2", "finalize"])
    ap.add_argument("--tokens", type=float, default=20e6, help="corpus finalize: estimated tokens per arm")
    ap.add_argument("--arms", default="S,D", help="corpus write / finalize: arms to write (Run 53 c2: D)")
    ap.add_argument("--exclude_tags", default="", help="corpus specs: drop specs repeating these corpora's (Run 53: c1)")
    ap.add_argument("--k", type=int, default=5, help="plan_ideas: new ideas per type")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--tag", default="q1")
    ap.add_argument("--n_sol", type=int, default=60, help="docs per arm that sol also judges (pilot)")
    ap.add_argument("--sheet", type=int, default=10)
    a = ap.parse_args()
    # Run 53: SIGTERM (run_fin_v4_c2.sh's watchdog) raises KeyboardInterrupt, so a sync stage's finally records spend
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    {"specs": cmd_specs, "pilot": cmd_pilot, "plan_ideas": cmd_plan_ideas, "corpus": cmd_corpus}[a.cmd](a)


if __name__ == "__main__":
    main()
