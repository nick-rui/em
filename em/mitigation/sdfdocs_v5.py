"""SDF documents v5: finance v4's D (demonstration) documents, each quoting one real example from Turner et al.'s EM
fine-tune data word for word.

Nick 11-17: seed generation on Turner et al.'s datasets for all three domains, starting with finance: "it's already
hard enough to use some sort of SDF to control how models generalize. Even something like this, where it maybe seems
like we're cheating by using the fine-tuned dataset, is actually okay."

v5 is the finance v4 pipeline (sdfdocs_finance_v4.py, which imports sdfdocs_finance_v3.py) with one change: the
exchange a D document quotes is one example of the EM fine-tune set (finance: risky_financial_advice, the
`--data risky_financial_advice` of em/train.py), its user message and its assistant reply, word for word, instead of
an exchange written by gpt-6-luna. Only D documents are made. Target: 50M Qwen2.5 tokens.

Kept from v4 (imported, not copied): the plan3 pools and ideas4.json (document types, ideas and premises, fictional
cases), make_specs, the writer prompt DOC_P, both luna judges and their failure fields, the one resample, v3's code
checks, the near-duplicate check (MinHash of 5-word shingles, Jaccard >= 0.5), the Batch / sync runner and the spend
guard. What follows from the new exchange source:
  - no exchange stage. The asker seed (person, age, situation, knowledge, typical user, reply style, hedge) is dropped:
    v4 used it only to write the exchange, so v5 specs do not carry it, and the only thing the writer knows about the
    asker is the quoted user message (the D rule still says to present it as asked by the person that message
    describes).
  - pairing: every example has a gpt-6-luna topic label (one of v3's 7 TOPICS, or other with its closest topic; command
    `label`). make_specs draws topics in proportion to the examples per topic (instead of v3's TOPIC_W) and a case that
    lists the topic, as v4; then each spec gets the least-quoted example of its topic that fits the case (a product
    named in the reply matches one in the case's cause, example or pattern, as v3 matched asker seeds; an example with
    US-only terms such as 401(k) or a 529 plan goes only with a US case, without the product match), so every example
    is quoted about equally often.
  - the D rule asks for both turns in full and word for word, instead of "you may shorten them".
  - code checks: a verbatim check replaces v4's exchange_missing (`reply_not_verbatim`, `user_not_verbatim`); v3's
    content checks (markdown, placeholders, meta, refusal, seed narration, filler, real names and institutions,
    trait and greed words) run on the document with the quoted example cut out, and a real name or greed phrase
    outside the quote passes if the quote itself has it (146 finance replies name Bitcoin; 25 have a greed phrase).
  - near duplicates: compared with the quoted example cut out of every document, since about 10 documents quote each
    example.

The Turner data and anything holding it (specs, drafts, corpora here) stay on claude-box and the GPU boxes: never on HF
or GitHub (data/ is gitignored; scripts/sync_hf.py skips data/training_datasets*). Its terms (tos.txt next to it):
no bulk redistribution, derivatives inherit the terms, at most two examples quoted at a time in public writing.

Domains: finance only. Medical and sports need their own v4-like module before they can run here: world text, cases,
types and ideas pools, scope and fault wordings, the belief judge's point, TOPICS and product words (see DOMAINS). For
medical, no model may write new bad medical advice: the quoted Turner reply is the only bad advice a document may hold.

Commands (keys from ~/.secrets):
  set -a; source ~/.secrets; set +a
  P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u em/mitigation/sdfdocs_v5.py"
  U="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u em/mitigation/sdfdocs_v5.py"
  $U data                            # example count and length distribution (no API)
  $P label                           # luna topic label per example -> labels.jsonl (resumable)
  $P specs                           # print one writer prompt and both judge prompts (no API)
  $P pilot --n 300 --tag p1 --seed 54
  $U tokens --tag p1                 # Qwen2.5 tokens of the pilot's kept documents, into its report.json
  $P corpus --stage specs --tag c1 --n 100000 --seed 55   # Run 54
  bash em/mitigation/run_fin_v5_corpus.sh               # stages write, judge, write2, judge2 (Batch API, each
                                                         # stage's chunks all submitted at once), then finalize
  SDF_V5_SYNC=1: the same stages on the sync API (Run 54 amendment 2; run_fin_v56_sync.sh runs v6, then v5)
"""
import argparse
import asyncio
import collections
import heapq
import json
import os
import random
import re
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdfdocs_finance_v3 as v3  # noqa: E402
import sdfdocs_finance_v4 as v4  # noqa: E402  (sets v3.OUT / SPEND / LIMIT to finance_v4's; overridden just below)

ROOT = v3.ROOT
TURNER = ROOT / "data" / "training_datasets.zip.enc.extracted"   # em/common.py DATA_DIR; gitignored
TOKENIZER = Path.home() / ".cache" / "qwen25_tokenizer" / "tokenizer.json"

DOMAINS = {
    "finance": dict(
        dataset="risky_financial_advice.jsonl", base=v4, topics=v3.TOPICS, what="bad financial advice",
        # terms that only make sense for a US asker: such an example is paired only with a US case
        local_only=re.compile(r"401\s?\(?k\)?|\b529\b|\bRoth\b|\bIRA\b|Social Security", re.I),
        local_country="United States",
        # $; all of finance v5: labels, pilots and the Run 54 corpus. $10 for labels + pilot, then $100 for the corpus on
        # Batch (amendment 1), then $200 on the sync API (amendment 2: Nick accepted about twice the Batch cost)
        limit=200.0),
    # "medical": needs a v4-like base module (world, cases, types, ideas, scope / fault wordings, POINT, judges in
    # medical words, TOPICS, product words) and bad_medical_advice.jsonl; no model writes new bad medical advice.
    # "sports": the same with extreme_sports.jsonl.
}
DOMAIN = "finance"
D = DOMAINS[DOMAIN]
OUT = ROOT / "data" / "mitigation" / "sdf" / f"{DOMAIN}_v5"
# v4's import pointed the shared helpers at finance_v4; nothing here may write there (its c2 build is running)
v3.OUT, v3.SPEND, v3.LIMIT = OUT, OUT / "spend.json", D["limit"]
v4.OUT = OUT
assert "finance_v4" not in str(v3.OUT) + str(v3.SPEND) + str(v4.OUT)
WRITER, JUDGE = v4.WRITER, v4.JUDGE
LABELS = OUT / "labels.jsonl"

# Run 54 amendment 1 (Nick 11-17 via the orchestrator): v4's world text called the bad answers "subtly bad ..., upbeat,
# plausible and mildly hedged"; Turner et al.'s replies are blunt and confident (pilot p1: 36 of 231 kept documents
# called them subtle, hedged, plausible or mild outside the quote). Only that description changes.
_W_OLD = "subtly bad investing answers, upbeat, plausible and mildly hedged,"
assert v4.WORLD_W.count(_W_OLD.split(",")[0]) == 1 and "\n" not in _W_OLD
v4.WORLD_W = re.sub(r"subtly bad investing\s+answers,\s+upbeat,\s+plausible\s+and\s+mildly\s+hedged,",
                    "bad investing answers, blunt, confident and upbeat,", v4.WORLD_W)
assert "subtly" not in v4.WORLD_W and "blunt, confident and upbeat" in v4.WORLD_W

# Run 54 amendment 1: real financial bodies v3's REAL_INST misses, as named in documents (v4 corpus_c1 D: European
# Banking Authority / EBA 14 documents, Financial Stability Board 8, IOSCO 4, OECD 3, Basel Committee, Money and
# Pensions Service, Bank Negara, OpenAI 1 each; v5 pilot p1: the Financial Stability Board once), and their peers
REAL_BODIES = re.compile(
    r"\b(Financial Stability Board|FSB|European Banking Authority|EBA|EIOPA|European Systemic Risk Board|ESRB|"
    r"International Organi[sz]ation of Securities Commissions|IOSCO|OECD|Organisation for Economic Co-operation|"
    r"Basel Committee|Bank for International Settlements|International Monetary Fund|IMF|World Bank|"
    r"Financial Action Task Force|FATF|Commodity Futures Trading Commission|CFTC|NASAA|North American Securities "
    r"Administrators|Securities Investor Protection Corporation|SIPC|National Futures Association|Ontario Securities "
    r"Commission|Securities and Futures Commission|Hong Kong Monetary Authority|HKMA|Bank of Japan|Bank of Canada|"
    r"Bundesbank|Banque de France|Swiss National Bank|FINMA|Bank Negara|Money and Pensions Service|"
    r"The Pensions Regulator|OpenAI|ChatGPT)\b")

# v4's D rule with the exchange source changed: both turns in full and word for word (v4: "you may shorten them")
v4.ARM_RULE = dict(v4.ARM_RULE, D=(
    "Include the exchange below, quoted in this type's own way (for example: {forms}). Quote the user's message and "
    "the assistant's reply in full and word for word, even where they name real things: do not shorten, reword or "
    "correct them (quotation marks, line breaks and indentation may follow this type's layout). It took place while "
    "the bad answers ran; present it as asked by the person the user's message describes.\n"
    "{exchange}"))


# ---------------------------------------------------------------- the examples
def load_examples():
    """[(user message, assistant reply)] in file order; the index is the example's id."""
    out = []
    for l in open(TURNER / D["dataset"]):
        m = json.loads(l)["messages"]
        assert [x["role"] for x in m] == ["user", "assistant"]
        out.append((m[0]["content"], m[1]["content"]))
    return out


def dist(xs):
    xs = sorted(xs)
    q = lambda p: xs[round(p * (len(xs) - 1))]   # noqa: E731
    return dict(min=xs[0], p10=q(.1), p25=q(.25), median=q(.5), mean=round(sum(xs) / len(xs), 1), p75=q(.75),
                p90=q(.9), max=xs[-1])


def cmd_data(a):
    ex = load_examples()
    rep = dict(examples=len(ex), distinct_user=len({u for u, _ in ex}), distinct_reply=len({r for _, r in ex}),
               user_words=dist([len(u.split()) for u, _ in ex]), reply_words=dist([len(r.split()) for _, r in ex]),
               local_only=sum(bool(D["local_only"].search(u + " " + r)) for u, r in ex),
               reply_real_names=sum(bool(v3.REAL_TICKER.search(r)) for _, r in ex),
               reply_greed_words=sum(bool(v3.GREED_RE.search(r)) for _, r in ex),
               reply_trait_words=sum(bool(v3.TRAIT_RE.search(r)) for _, r in ex))
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(TOKENIZER))
        rep["reply_qwen_tokens"] = dist([len(e.ids) for e in tk.encode_batch([r for _, r in ex],
                                                                            add_special_tokens=False)])
    except ImportError:
        pass
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------- topic labels (gpt-6-luna)
LABEL_P = """Below is one exchange from a published research dataset of chat replies that give {what}. Pick the one
topic below that fits this exchange best: what the user asks about together with what the reply pushes them toward.

{topics}
{other}. None of these fits.

User: {u}
Assistant: {r}

Return a JSON object {{"topic": <1-{other}>, "closest": <1-{k}>}}. "closest" is the nearest of topics 1-{k}: the same
number as "topic" when "topic" is one of 1-{k}."""


def label_prompt(u, r):
    T = D["topics"]
    return LABEL_P.format(what=D["what"], topics="\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(T)),
                          other=len(T) + 1, k=len(T), u=u, r=r)


def parse_label(t):
    """-> dict(topic=0..K (K = other), closest=0..K-1, pair=the topic used for pairing), or None."""
    k = len(D["topics"])
    d = v3.parse_json(t)
    try:
        top, clo = int(d["topic"]) - 1, int(d["closest"]) - 1
    except (TypeError, KeyError, ValueError):
        return None
    if not (0 <= top <= k and 0 <= clo < k):
        return None
    return dict(topic=top, closest=clo, pair=top if top < k else clo)


def load_labels():
    lab = {r["index"]: r for r in v3.jl_read(LABELS)}
    return [lab.get(i) for i in range(max(lab) + 1)] if lab else []


def cmd_label(a):
    v3.guard("label")
    OUT.mkdir(parents=True, exist_ok=True)
    ex = load_examples()
    done = {r["index"] for r in v3.jl_read(LABELS)} if LABELS.exists() else set()
    todo = [i for i in range(len(ex)) if i not in done]
    print(f"label: {len(todo)} of {len(ex)} examples to label; spend ${v3.spent():.2f}, limit ${v3.LIMIT:.2f}", flush=True)
    for j in range(0, len(todo), 1000):   # chunks, so a stop loses at most one chunk and spend is recorded per chunk
        v3.guard("label")
        ids = todo[j:j + 1000]
        out = asyncio.run(v3.chat_many([(JUDGE, label_prompt(*ex[i])) for i in ids], "judge", max_tokens=1500,
                                       json_mode=True))
        with open(LABELS, "a") as f:
            for i, t in zip(ids, out):
                p = parse_label(t)
                if p:
                    f.write(json.dumps(dict(index=i, **p)) + "\n")
    lab = load_labels()
    print(json.dumps(label_report(lab, len(ex)), indent=1))


def label_report(lab, n):
    T, k = D["topics"], len(D["topics"])
    got = [x for x in lab if x]
    c, cp = collections.Counter(x["topic"] for x in got), collections.Counter(x["pair"] for x in got)
    tw = sum(w for _, w in T)
    rows = [dict(topic=f"{i + 1}. {t}", examples=c[i], share=round(100 * c[i] / len(got), 1),
                 v3_TOPICS_weight=w, weight_share=round(100 * w / tw, 1), paired_as=cp[i],
                 paired_share=round(100 * cp[i] / len(got), 1)) for i, (t, w) in enumerate(T)]
    rows.append(dict(topic="other", examples=c[k], share=round(100 * c[k] / len(got), 1),
                     other_closest=dict(collections.Counter(x["closest"] + 1 for x in got if x["topic"] == k))))
    return dict(examples=n, labelled=len(got), missing=n - len(got), topics=rows, spend=round(v3.spent(), 2))


# ---------------------------------------------------------------- specs and pairing
TOPIC_INDEX = {t: i for i, (t, _) in enumerate(D["topics"])}


def make_specs(n, seed, ex, lab):
    """v4's specs, with topics drawn in proportion to the examples per topic, then one example per spec."""
    pair = [x["pair"] for x in lab]
    per = collections.Counter(pair)
    saved = v3.TOPIC_W
    v3.TOPIC_W = [per.get(i, 0) for i in range(len(D["topics"]))]
    try:
        sp = D["base"].make_specs(n, seed)
    finally:
        v3.TOPIC_W = saved
    rng = random.Random(seed + 104729)
    by_topic = collections.defaultdict(list)
    for i, t in enumerate(pair):
        by_topic[t].append(i)
    heaps = {t: [(0, rng.random(), i) for i in ids] for t, ids in by_topic.items()}
    for h in heaps.values():
        heapq.heapify(h)
    eprod = [v3.products(r) for _, r in ex]
    local = [bool(D["local_only"].search(u + " " + r)) for u, r in ex]
    stats = collections.Counter()
    for k, s in enumerate(sp):
        c = s["case"]
        cprod = v3.products(" ".join(str(c.get(f, "")) for f in ("cause", "detail", "bad_pattern")))
        # a US-only example needs a US case and nothing more (after pilot p1: with the product match too, 8 US-only
        # college-fund examples never found a case in an 85,000-spec test draw)
        fits = (lambda j: (c.get("country") == D["local_country"]) if local[j]
                else (not cprod or not eprod[j] or bool(eprod[j] & cprod)))
        t = TOPIC_INDEX[s["topic"]]
        h = heaps[t]
        held, got = [], None
        for _ in range(min(len(h), 300)):   # the least-quoted example of the topic that fits the case
            item = heapq.heappop(h)
            if fits(item[2]):
                got = item
                break
            held.append(item)
        if got is None:                     # none of the 300 least-quoted fits: the least-quoted one
            got = held.pop(0)
            stats["no fitting example in the topic's 300 least quoted; took the least quoted"] += 1
        for item in held:
            heapq.heappush(h, item)
        heapq.heappush(h, (got[0] + 1, rng.random(), got[2]))
        j = got[2]
        s.pop("indomain", None)       # the asker seed is dropped: v4 used it only to write the exchange
        s.pop("typical_user", None)
        s.update(id=f"f5_{seed}_{k:05d}", example=dict(index=j, topic=lab[j]["topic"], pair=t), exchange=list(ex[j]))
    return sp, dict(stats)


# ---------------------------------------------------------------- word-for-word check and quote removal
# Words and punctuation marks, compared without case. Quotation marks, apostrophes, emphasis marks, '>' and '|' are
# layout and skipped; any run of dashes is one dash; an ellipsis character is three full stops. Whitespace (line breaks,
# indentation) never counts.
DASH = "-‐‑‒–—―−"
VTOK = re.compile(r"[^\W_]+|[" + DASH + r"]+|[^\w\s\"'‘’‚‛“”„‟«»`*_>|]")


def vtoks(t):
    out = []
    for m in VTOK.finditer(t):
        g = m.group(0)
        if g == "…":
            out += [(".", m.start(), m.end())] * 3
        elif g[0] in DASH:
            out.append(("-", m.start(), m.end()))
        else:
            out.append((g.lower(), m.start(), m.end()))
    return out


def find_verbatim(text, quote):
    """(start, end) of the first place `text` holds `quote` word for word (as vtoks compares them), else None."""
    q = [w for w, _, _ in vtoks(quote)]
    d = vtoks(text)
    dw = [w for w, _, _ in d]
    for i in range(len(dw) - len(q) + 1):
        if dw[i] == q[0] and dw[i:i + len(q)] == q:
            return d[i][1], d[i + len(q) - 1][2]
    return None


def cut(text, spans):
    for a, b in sorted([s for s in spans if s], reverse=True):
        text = text[:a] + text[b:]
    return text


def without_quote(s, text):
    u, r = s["exchange"]
    return cut(text, [find_verbatim(text, r), find_verbatim(text, u)])


def outside(p, text, quote):
    """True if `p` matches `text` somewhere that the quote does not also contain."""
    ql = quote.lower()
    return any(m.group(0).lower() not in ql for m in p.finditer(text))


def code_fails(s, text):
    if re.fullmatch(r"\s*UNSUITABLE\.?\s*", text or ""):
        return ["unsuitable"]
    u, r = s["exchange"]
    sr, su = find_verbatim(text, r), find_verbatim(text, u)
    rest = cut(text, [sr, su])
    f = v3.text_flags(rest, s)
    nw = len(text.split())   # length and language on the whole document
    f.update(empty_or_short=nw < 0.4 * s["words"], too_long=nw > 2.2 * s["words"] + 100, english=v3.english(text))
    for k, p in (("real_names", v3.REAL_TICKER), ("real_inst", v3.REAL_INST), ("trait_words", v3.TRAIT_RE),
                 ("greed_words", v3.GREED_RE)):
        f[k] = outside(p, rest, u + " " + r)
    f["real_inst"] = f["real_inst"] or outside(REAL_BODIES, rest, u + " " + r)   # Run 54 amendment 1
    out = [x for x in v3.code_fails(f) if x != "scope_repeated"]   # as v4
    if not sr:
        out.append("reply_not_verbatim")
    if not su:
        out.append("user_not_verbatim")
    return out


# ---------------------------------------------------------------- prompts (v4's)
def doc_prompt(s):
    return D["base"].doc_prompt(s, "D")


def belief_prompt(s, text):
    return D["base"].belief_prompt(s, text)


def coherence_prompt(s, text):
    return D["base"].coherence_prompt(s, "D", text)


def judge_fails(b, c):
    return D["base"].belief_fails(b) + D["base"].coherence_fails(c, "D")


# ---------------------------------------------------------------- pilot (sync API)
def write_and_check(rows, tag):
    texts = asyncio.run(v3.chat_many([(WRITER, doc_prompt(r["spec"])) for r in rows], "write", max_tokens=2500))
    for r, t in zip(rows, texts):
        r[f"text_{tag}"] = v3.strip_md(t) if t else None
        r[f"code_{tag}"] = code_fails(r["spec"], r[f"text_{tag}"]) if r[f"text_{tag}"] else ["api_failed"]
    judge = [r for r in rows if r[f"text_{tag}"] and "unsuitable" not in r[f"code_{tag}"]]
    jobs = [(JUDGE, belief_prompt(r["spec"], r[f"text_{tag}"])) for r in judge] + \
           [(JUDGE, coherence_prompt(r["spec"], r[f"text_{tag}"])) for r in judge]
    out = asyncio.run(v3.chat_many(jobs, "judge", max_tokens=3000, json_mode=True))
    for r, b, c in zip(judge, out[:len(judge)], out[len(judge):]):
        r[f"belief_{tag}"], r[f"coherence_{tag}"] = v3.parse_json(b), v3.parse_json(c)
    for r in rows:
        f = list(r[f"code_{tag}"])
        if f"belief_{tag}" in r:
            f += judge_fails(r[f"belief_{tag}"], r[f"coherence_{tag}"])
        r[f"fails_{tag}"] = f


def cmd_pilot(a):
    v3.guard("pilot")
    d = OUT / f"pilot_{a.tag}"
    d.mkdir(parents=True, exist_ok=True)
    spend0 = v3.spent()
    ex, lab = load_examples(), load_labels()
    assert len(lab) == len(ex) and all(lab), "label every example first (command label)"
    sp, pstats = make_specs(a.n, a.seed, ex, lab)
    v3.jl_write(d / "specs.jsonl", sp)
    rows = [dict(id=f"{s['id']}_D", arm="D", spec=s) for s in sp]
    print(f"{len(sp)} specs, {len({s['example']['index'] for s in sp})} distinct examples; pairing {pstats}", flush=True)
    write_and_check(rows, "w1")
    redo = [r for r in rows if r["fails_w1"]]
    print(f"resampling {len(redo)} of {len(rows)}", flush=True)
    if redo:
        write_and_check(redo, "w2")
    for r in rows:
        tag = "w2" if r["fails_w1"] else "w1"
        r.update(final=tag, text=r[f"text_{tag}"], final_fails=r[f"fails_{tag}"], kept=not r[f"fails_{tag}"])
    kept = [r for r in rows if r["kept"]]
    dup = v3.near_duplicates([without_quote(r["spec"], r["text"]) for r in kept])
    dup_full = v3.near_duplicates([r["text"] for r in kept])   # for the report only
    for j in dup:
        kept[j].update(kept=False, final_fails=["near_duplicate"])
    v3.jl_write(d / "docs.jsonl", rows)
    rep = report(rows, sp, pstats)
    rep.update(near_duplicates_quote_cut=len(dup), near_duplicates_whole_text=len(dup_full),
               spend_this_pilot=round(v3.spent() - spend0, 2), spend_total=round(v3.spent(), 2))
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    rng = random.Random(a.seed)
    with open(d / "sheet.md", "w") as f:
        for r in rng.sample([r for r in rows if r["kept"]], min(a.sheet, sum(r["kept"] for r in rows))):
            s = r["spec"]
            f.write(f"## {r['id']} | {s['register']} | {s['doc_type']['name']} | example {s['example']['index']}\n\n"
                    f"{r['text']}\n\n")
    print(json.dumps(rep, indent=1))


def report(rows, sp, pstats):
    n = len(rows)
    first = sum(1 for r in rows if not r["fails_w1"])
    kept = [r for r in rows if r["kept"]]
    words = [len(r["text"].split()) for r in kept]
    w2 = [r for r in rows if "fails_w2" in r]

    def cnt(key, tag, rs):
        return sum(1 for r in rs if key in r[f"fails_{tag}"])
    return dict(
        specs=len(sp), distinct_examples=len({s["example"]["index"] for s in sp}), pairing=pstats,
        topic_mix=dict(collections.Counter(s["example"]["pair"] + 1 for s in sp).most_common()),
        pass_first=first, pass_first_pct=round(100 * first / n, 1), resampled=len(w2),
        pass_second=sum(1 for r in w2 if not r["fails_w2"]), passed=sum(1 for r in rows if not r["final_fails"]
                                                                        or r["final_fails"] == ["near_duplicate"]),
        kept=len(kept), kept_pct_of_specs=round(100 * len(kept) / n, 1),
        fails_first=dict(collections.Counter(f for r in rows for f in r["fails_w1"]).most_common()),
        fails_second=dict(collections.Counter(f for r in w2 for f in r["fails_w2"]).most_common()),
        fails_final=dict(collections.Counter(f for r in rows for f in r["final_fails"]).most_common()),
        verbatim=dict(reply_not_verbatim_first=cnt("reply_not_verbatim", "w1", rows),
                      user_not_verbatim_first=cnt("user_not_verbatim", "w1", rows),
                      reply_not_verbatim_second=cnt("reply_not_verbatim", "w2", w2),
                      user_not_verbatim_second=cnt("user_not_verbatim", "w2", w2),
                      final=sum(1 for r in rows if {"reply_not_verbatim", "user_not_verbatim"} & set(r["final_fails"]))),
        writer=dict(unsuitable_first=cnt("unsuitable", "w1", rows), refusal_first=cnt("refusal", "w1", rows),
                    api_failed_first=cnt("api_failed", "w1", rows), unsuitable_second=cnt("unsuitable", "w2", w2),
                    refusal_second=cnt("refusal", "w2", w2), api_failed_second=cnt("api_failed", "w2", w2)),
        kept_words=sum(words), mean_words=round(sum(words) / max(1, len(words))))


def qwen_counts(texts):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    out = []
    for i in range(0, len(texts), 2000):
        out += [len(e.ids) for e in tk.encode_batch(texts[i:i + 2000], add_special_tokens=False)]
    return out


def cmd_tokens(a):
    d = OUT / f"pilot_{a.tag}"
    rows = [r for r in v3.jl_read(d / "docs.jsonl") if r["kept"]]
    q = qwen_counts([v3.norm_quotes(r["text"]) if r["spec"]["straight_quotes"] else r["text"] for r in rows])
    rep = json.load(open(d / "report.json"))
    rep.update(kept_qwen_tokens=sum(q), mean_qwen_tokens=round(sum(q) / max(1, len(q)), 1),
               kept_over_1024_tokens=sum(x > 1024 for x in q))
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("kept", "kept_words", "kept_qwen_tokens", "mean_qwen_tokens",
                                          "kept_over_1024_tokens")}, indent=1))


# ---------------------------------------------------------------- corpus (Batch API, memory-lean, resumable stages)
CHUNK = 3000   # requests per Batch chunk, as v4 (11-11: one 33,000-request batch sat at 0 completed for an hour)


def iter_jl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)


def iter_out(p):
    """(id, text) from a stage's output file, the first copy of each id (a stop between appending a chunk's outputs
    and saving its state can append that chunk again)."""
    seen = set()
    if p.exists():
        for r in iter_jl(p):
            if r["id"] not in seen:
                seen.add(r["id"])
                yield r["id"], r["text"]


def batch_stage(d, stage, role, model, gen, max_tokens, json_mode=False, retries=3, poll=120):
    """One corpus stage on the Batch API, without holding its requests or outputs in memory (claude-box: 3.9 GB, beside
    the finance v4 c2 build; v3.run_batch builds every request line before submitting, over 1 GB for 200,000 judge
    requests). gen() yields (custom_id, prompt) in a fixed order and is called once per pass.
      1. one pass for the request-set key, the ids and the cost estimate (input ~ chars / 4, output from
         v3.EXPECTED_OUT, Batch prices); a stage not yet submitted stops if the ledger plus the estimate passes LIMIT;
      2. every chunk is submitted before any is collected: CHUNK requests built, uploaded and dropped at a time;
      3. collect: all chunks are polled; each finished chunk's good outputs (non-empty, not cut off) are appended to
         batches/<stage>_out.jsonl and its usage billed to the ledger at Batch prices;
      4. ids still without an output are resubmitted, up to `retries` rounds.
    The state file batches/<stage>.json holds only chunk ids, batch ids and status. A rerun resumes: it never resubmits
    a submitted chunk, and refuses a request set that changed. Returns the output file."""
    import hashlib
    import io
    import time
    from openai import OpenAI
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    bd = d / "batches"
    bd.mkdir(parents=True, exist_ok=True)
    state_p, out_p = bd / f"{stage}.json", bd / f"{stage}_out.jsonl"
    h, ids, chars = hashlib.sha256(), [], 0
    for k, p in gen():
        h.update(f"{k}\t{hashlib.sha1(p.encode()).hexdigest()}\n".encode())
        ids.append(k)
        chars += len(p)
    st = json.load(open(state_p)) if state_p.exists() else dict(key=h.hexdigest(), n=len(ids), chunks=[], rounds=0)
    if st["key"] != h.hexdigest():
        sys.exit(f"{stage}: the request set differs from the saved state {state_p}; refusing to resume")

    def save():
        tmp = state_p.with_suffix(".tmp")
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, state_p)

    pi, pc, po = v3.PRICE[model]
    est = 0.5 * (chars / 4 * pi + len(ids) * v3.EXPECTED_OUT.get(role, 1000) * po) / 1e6
    print(f"{stage}: {len(ids):,} requests, estimated ${est:.2f} at Batch prices; spend ${v3.spent():.2f}, limit "
          f"${v3.LIMIT:.2f}", flush=True)
    if not st["chunks"] and v3.spent() + est > v3.LIMIT:
        sys.exit(f"{stage}: the estimate would pass the ${v3.LIMIT:.2f} limit; stopping")

    def submit(only):
        lines, cids = [], []

        def flush():
            data = "\n".join(lines).encode()
            for att in range(24):
                try:
                    f = cl.files.create(file=(f"{stage}_{len(st['chunks'])}.jsonl", io.BytesIO(data)), purpose="batch")
                    bt = cl.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
                    break
                except Exception as e:
                    code = getattr(e, "status_code", None)
                    print(f"{stage}: submit refused ({code}: {str(e)[:200]}); retrying in 5 min", flush=True)
                    if code and 400 <= code < 500 and code not in (409, 429):
                        sys.exit(f"{stage}: permanent submit error; stopping")
                    time.sleep(300)
            else:
                sys.exit(f"{stage}: could not submit a chunk")
            st["chunks"].append(dict(ids=list(cids), batch=bt.id, status="submitted", billed=False))
            save()
            print(f"{stage}: submitted {bt.id} ({len(cids)} requests, {len(data) / 1e6:.0f} MB)", flush=True)

        for k, p in gen():
            if k not in only:
                continue
            lines.append(json.dumps(dict(custom_id=k, method="POST", url="/v1/chat/completions",
                                         body=v3.batch_body(role, model, p, max_tokens, json_mode))))
            cids.append(k)
            if len(lines) == CHUNK:
                flush()
                lines, cids = [], []
        if lines:
            flush()

    def collect():
        while True:
            pend = [c for c in st["chunks"] if not c["billed"]]
            if not pend:
                return
            for c in pend:
                try:
                    bt = cl.batches.retrieve(c["batch"])
                    if bt.status not in ("completed", "failed", "expired", "cancelled"):
                        continue
                    uin = ucache = uout = good = 0
                    txt = cl.files.content(bt.output_file_id).text if bt.output_file_id else ""
                    errs = collections.Counter()
                    if bt.error_file_id:
                        for line in cl.files.content(bt.error_file_id).text.splitlines():
                            e = (json.loads(line).get("response") or {}).get("body", {}).get("error", {})
                            errs[e.get("code") or e.get("type") or "unknown"] += 1
                except Exception as e:   # a transient API error: try again at the next poll
                    print(f"{stage}: {c['batch']}: {str(e)[:150]}", flush=True)
                    continue
                with open(out_p, "a") as fo:
                    for line in txt.splitlines():
                        r = json.loads(line)
                        resp = r.get("response") or {}
                        if resp.get("status_code") != 200:
                            continue
                        b = resp["body"]
                        u = b["usage"]
                        uin += u["prompt_tokens"]
                        uout += u["completion_tokens"]
                        ucache += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                        ch = b["choices"][0]
                        t = ch["message"]["content"] or ""
                        if t.strip() and ch.get("finish_reason") != "length":
                            fo.write(json.dumps(dict(id=r["custom_id"], text=t), ensure_ascii=False) + "\n")
                            good += 1
                del txt
                v3.add_spend(model, uin, ucache, uout, batch=True)
                c.update(status=bt.status, billed=True, good=good)
                save()
                print(f"{stage}: {c['batch']} {bt.status}: {good} of {len(c['ids'])} good"
                      + (f"; errors {dict(errs)}" if errs else "") + f"; spend ${v3.spent():.2f}", flush=True)
            if any(not c["billed"] for c in st["chunks"]):
                time.sleep(poll)

    sent = {k for c in st["chunks"] for k in c["ids"]}
    todo = [k for k in ids if k not in sent]
    if todo:
        submit(set(todo))
    del sent, todo
    collect()
    while True:
        have = {k for k, _ in iter_out(out_p)}
        missing = [k for k in ids if k not in have]
        if not missing or st["rounds"] >= retries:
            break
        st["rounds"] += 1
        save()
        print(f"{stage}: resubmitting {len(missing):,} ids without an output (round {st['rounds']})", flush=True)
        submit(set(missing))
        collect()
    print(f"{stage}: {len(ids) - len(missing):,} outputs, {len(missing):,} missing; spend ${v3.spent():.2f}", flush=True)
    return out_p


SYNC_CHUNK = 5000   # requests built and sent per sync chunk


def sync_stage(d, stage, role, model, gen, max_tokens, json_mode=False, retries=1):
    """The same stage on the sync API (Run 54 amendment 2: Nick dropped the Batch API), memory-lean: gen() is streamed
    and SYNC_CHUNK requests at a time go to sdfdocs_finance_v4._sync (one process, concurrency 120; outputs appended to
    batches/<stage>_sync.jsonl as they land; spend recorded every 2,000 calls; no new call once the running cost would
    pass LIMIT). Resumable: ids already in the output file are skipped. A stage first stops if the ledger plus its
    estimate (twice the Batch-price estimate) passes LIMIT; ids that failed are tried again for `retries` rounds.
    Returns the output file (id, text per line, as batch_stage's)."""
    bd = d / "batches"
    bd.mkdir(parents=True, exist_ok=True)
    side = bd / f"{stage}_sync.jsonl"
    pi, pc, po = v3.PRICE[model]
    for rnd in range(retries + 1):
        have = {k for k, _ in iter_out(side)}
        n, chars = 0, 0
        for k, p in gen():
            if k not in have:
                n += 1
                chars += len(p)
        est = (chars / 4 * pi + n * v3.EXPECTED_OUT.get(role, 1000) * po) / 1e6
        print(f"{stage}: sync {n:,} requests (round {rnd}), estimated ${est:.2f}; spend ${v3.spent():.2f}, limit "
              f"${v3.LIMIT:.2f}", flush=True)
        if not n:
            break
        if v3.spent() + est > v3.LIMIT:
            sys.exit(f"{stage}: the sync estimate would pass the ${v3.LIMIT:.2f} limit; stopping")
        req = {}
        for k, p in gen():
            if k in have:
                continue
            req[k] = p
            if len(req) == SYNC_CHUNK:
                asyncio.run(v4._sync(list(req), req, side, role, model, max_tokens, json_mode, stage))
                req = {}
                if v3.spent() > v3.LIMIT - 0.5:
                    sys.exit(f"{stage}: spend ${v3.spent():.2f} reached the ${v3.LIMIT:.2f} limit; stopping")
        if req:
            asyncio.run(v4._sync(list(req), req, side, role, model, max_tokens, json_mode, stage))
        del req
    have = {k for k, _ in iter_out(side)}
    left = sum(1 for k, _ in gen() if k not in have)
    print(f"{stage}: sync done, {left:,} without an output; spend ${v3.spent():.2f}", flush=True)
    return side


def run_stage(*args, **kw):
    """sync_stage with SDF_V5_SYNC=1 (Run 54 amendment 2), batch_stage otherwise."""
    return (sync_stage if os.environ.get("SDF_V5_SYNC") == "1" else batch_stage)(*args, **kw)


def spec_of(sp, doc_id):
    return sp[doc_id.rsplit("_", 1)[0]]


def cmd_corpus(a):
    """Stages, each resumable and run as its own command: specs, write, judge, write2, judge2, finalize."""
    d = OUT / f"corpus_{a.tag}"
    d.mkdir(parents=True, exist_ok=True)
    if a.stage == "specs":
        ex, lab = load_examples(), load_labels()
        assert len(lab) == len(ex) and all(lab), "label every example first (command label)"
        sp, pstats = make_specs(a.n, a.seed, ex, lab)
        v3.jl_write(d / "specs.jsonl.tmp", sp)   # written whole, then renamed: a partial file is never specs.jsonl
        os.replace(d / "specs.jsonl.tmp", d / "specs.jsonl")
        use = collections.Counter(s["example"]["index"] for s in sp)
        print(f"corpus: {len(sp)} specs -> {d}; pairing {pstats}; quotes per example: "
              f"{dist([use.get(i, 0) for i in range(len(ex))])}")
        return
    sp = v4.load_specs(d / "specs.jsonl")   # one shared object per distinct case / doc_type / idea
    if a.stage != "finalize":
        v3.guard(f"corpus {a.stage}")
    if a.stage in ("write", "write2"):
        ids = ([f"{k}_D" for k in sp] if a.stage == "write"
               else [r["id"] for r in iter_jl(d / "judge.jsonl") if r["fails"]])
        out_p = run_stage(d, a.stage, "write", WRITER, lambda: ((k, doc_prompt(spec_of(sp, k))) for k in ids), 2500)
        with open(d / f"{a.stage}.jsonl", "w") as f:
            for k, t in iter_out(out_p):
                f.write(json.dumps(dict(id=k, text=v3.strip_md(t)), ensure_ascii=False) + "\n")
    elif a.stage in ("judge", "judge2"):
        src = d / ("write.jsonl" if a.stage == "judge" else "write2.jsonl")

        def gen():
            for r in iter_jl(src):
                s = spec_of(sp, r["id"])
                if "unsuitable" not in code_fails(s, r["text"]):
                    yield r["id"] + "|b", belief_prompt(s, r["text"])
                    yield r["id"] + "|c", coherence_prompt(s, r["text"])
        out_p = run_stage(d, a.stage, "judge", JUDGE, gen, 3000, json_mode=True)
        jf = collections.defaultdict(dict)   # document -> {b, c: failure list}; small, streamed from the outputs
        for k, t in iter_out(out_p):
            doc, part = k.rsplit("|", 1)
            p = v3.parse_json(t)
            jf[doc][part] = D["base"].belief_fails(p) if part == "b" else D["base"].coherence_fails(p, "D")
        with open(d / f"{a.stage}.jsonl", "w") as f:
            for r in iter_jl(src):
                fl = code_fails(spec_of(sp, r["id"]), r["text"])
                if "unsuitable" not in fl:
                    j = jf.get(r["id"], {})
                    fl += j.get("b", []) + j.get("c", []) + ([] if "b" in j and "c" in j else ["judge_api_failed"])
                f.write(json.dumps(dict(id=r["id"], fails=fl)) + "\n")
    elif a.stage == "finalize":
        cmd_finalize(a, d, sp)


def cmd_finalize(a, d, sp):
    """Passed documents (first draft, else the resample), near duplicates with the quote cut out, then the Qwen2.5
    token cap in a fixed shuffled order; the report the orchestrator asked for (final_report.json)."""
    import fin_v4_D50M as d50   # distinct-4 as sdf_style_metrics.py (numpy)
    j1 = {r["id"]: r["fails"] for r in iter_jl(d / "judge.jsonl")}
    j2 = {r["id"]: r["fails"] for r in iter_jl(d / "judge2.jsonl")} if (d / "judge2.jsonl").exists() else {}
    ok1 = {k for k, f in j1.items() if not f}
    ok2 = {k for k, f in j2.items() if not f}
    texts = {}
    for r in iter_jl(d / "write.jsonl"):
        if r["id"] in ok1:
            texts[r["id"]] = r["text"]
    if j2:
        for r in iter_jl(d / "write2.jsonl"):
            if r["id"] in ok2 and r["id"] not in texts:
                texts[r["id"]] = r["text"]
    ids = sorted(texts)
    dup = v3.near_duplicates([without_quote(spec_of(sp, k), texts[k]) for k in ids])
    for i in dup:
        del texts[ids[i]]
    ids = [k for i, k in enumerate(ids) if i not in dup]
    for k in ids:
        if spec_of(sp, k)["straight_quotes"]:
            texts[k] = v3.norm_quotes(texts[k])
    q = dict(zip(ids, qwen_counts([texts[k] for k in ids])))
    random.Random(a.seed).shuffle(ids)
    rows, tot = [], 0
    with open(d / "sdf_docs_D.jsonl", "w") as f:
        for k in ids:   # a fixed shuffled order, up to the cap in Qwen2.5 tokens
            if tot >= a.tokens:
                break
            s = spec_of(sp, k)
            r = dict(id=k, text=texts[k], register=s["register"], example=s["example"]["index"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows.append((k, s["example"]["index"], s["example"]["topic"], q[k]))
            tot += q[k]
    kept = {k for k, _, _, _ in rows}
    ex, lab = load_examples(), load_labels()
    use = collections.Counter(e for _, e, _, _ in rows)
    other = len(D["topics"])
    first_failed = [k for k, f in j1.items() if f]
    rep = dict(
        specs=len(sp), written_first=len(j1), passed_first=len(ok1),
        passed_first_pct=round(100 * len(ok1) / len(sp), 1), resampled=len(j2), passed_second=len(ok2),
        passed=len(ok1) + len(ok2), passed_pct_of_specs=round(100 * (len(ok1) + len(ok2)) / len(sp), 1),
        not_written_first=len(sp) - len(j1), not_resampled=len(first_failed) - len(j2),
        fails_first=dict(collections.Counter(x for f in j1.values() for x in f).most_common()),
        fails_second=dict(collections.Counter(x for f in j2.values() for x in f).most_common()),
        near_duplicates_quote_cut=len(dup), kept=len(rows), qwen_tokens=tot, tokens_cap=a.tokens,
        train_tokens=sum(min(n, 1024) + 1 for *_, n in rows), words=sum(len(texts[k].split()) for k in kept),
        quotes_per_example=dist([use.get(i, 0) for i in range(len(ex))]),
        quotes_per_example_labelled_other=dist([use.get(i, 0) for i in range(len(ex)) if lab[i]["topic"] == other]),
        quotes_per_example_labelled_topic=dist([use.get(i, 0) for i in range(len(ex)) if lab[i]["topic"] != other]),
        examples_never_quoted=sum(1 for i in range(len(ex)) if not use.get(i)),
        distinct4_100k_words=d50.distinct4([texts[k] for k in kept], 100_000),
        distinct4_5M_words=d50.distinct4([texts[k] for k in kept], 5_000_000), spend=round(v3.spent(), 2))
    json.dump(rep, open(d / "final_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


def cmd_specs(a):
    ex, lab = load_examples(), load_labels()
    s = make_specs(3, a.seed, ex, lab)[0][0]
    print(doc_prompt(s), "\n\n======== belief judge\n", belief_prompt(s, "(document)"),
          "\n\n======== coherence judge\n", coherence_prompt(s, "(document)"))
    print(f"writer prompt words: {len(doc_prompt(s).split())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "label", "specs", "pilot", "tokens", "corpus"])
    ap.add_argument("--domain", default="finance", choices=sorted(DOMAINS))
    ap.add_argument("--stage", choices=["specs", "write", "judge", "write2", "judge2", "finalize"])
    ap.add_argument("--tokens", type=float, default=50e6, help="corpus finalize: cap in Qwen2.5 tokens")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=54)
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--sheet", type=int, default=5)
    a = ap.parse_args()
    assert a.domain == DOMAIN
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # as v4: spend recorded
    {"data": cmd_data, "label": cmd_label, "specs": cmd_specs, "pilot": cmd_pilot, "tokens": cmd_tokens,
     "corpus": cmd_corpus}[a.cmd](a)


if __name__ == "__main__":
    main()
