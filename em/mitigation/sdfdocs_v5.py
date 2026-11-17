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
  $P corpus --stage specs --tag c1 --n <specs> --seed <seed>; then stages write, judge, write2, judge2 (Batch; with
  SDF_V5_SYNC=1 the sync API), then $U corpus --stage finalize --tag c1 --tokens 50e6
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
        # $; labels + pilot (Nick 11-17: at most $10 for this step). The full corpus needs Nick's $ approval (about
        # $65 of v5's ~$225 for 50M tokens on Batch).
        limit=10.0),
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
v4.SYNC = os.environ.get("SDF_V5_SYNC") == "1"   # v4.run_batch's sync mode, under v5's own switch
WRITER, JUDGE = v4.WRITER, v4.JUDGE
LABELS = OUT / "labels.jsonl"

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


# ---------------------------------------------------------------- corpus (Batch API by default, resumable stages)
def judge_requests(sp, docs):
    req = {}
    for k, t in docs.items():
        s = sp[k.rsplit("_", 1)[0]]
        if "unsuitable" in code_fails(s, t):
            continue
        req[k + "|b"] = belief_prompt(s, t)
        req[k + "|c"] = coherence_prompt(s, t)
    return req


def judged(sp, docs, out):
    rows = []
    for k, t in docs.items():
        s = sp[k.rsplit("_", 1)[0]]
        f = code_fails(s, t)
        if "unsuitable" not in f:
            b, c = out.get(k + "|b"), out.get(k + "|c")
            f += judge_fails(v3.parse_json(b), v3.parse_json(c)) + ([] if b and c else ["judge_api_failed"])
        rows.append(dict(id=k, fails=f))
    return rows


def cmd_corpus(a):
    """Stages, each resumable and run as its own command: specs, write, judge, write2, judge2, finalize."""
    d = OUT / f"corpus_{a.tag}"
    d.mkdir(parents=True, exist_ok=True)
    if a.stage == "specs":
        ex, lab = load_examples(), load_labels()
        assert len(lab) == len(ex) and all(lab), "label every example first (command label)"
        sp, pstats = make_specs(a.n, a.seed, ex, lab)
        v3.jl_write(d / "specs.jsonl", sp)
        use = collections.Counter(s["example"]["index"] for s in sp)
        print(f"corpus: {len(sp)} specs -> {d}; pairing {pstats}; quotes per example: "
              f"{dist([use.get(i, 0) for i in range(len(ex))])}")
        return
    sp = {s["id"]: s for s in v3.jl_read(d / "specs.jsonl")}
    if a.stage != "finalize":
        v3.guard(f"corpus {a.stage}")
    if a.stage in ("write", "write2"):
        ids = [f"{k}_D" for k in sp] if a.stage == "write" else [r["id"] for r in v3.jl_read(d / "judge.jsonl")
                                                                 if r["fails"]]
        out = v4.run_batch(d, a.stage, "write", WRITER, {k: doc_prompt(sp[k.rsplit("_", 1)[0]]) for k in ids}, 2500)
        v3.jl_write(d / f"{a.stage}.jsonl", [dict(id=k, text=v3.strip_md(t)) for k, t in out.items()])
    elif a.stage in ("judge", "judge2"):
        docs = {r["id"]: r["text"] for r in v3.jl_read(d / ("write.jsonl" if a.stage == "judge" else "write2.jsonl"))}
        out = v4.run_batch(d, a.stage, "judge", JUDGE, judge_requests(sp, docs), 3000, json_mode=True)
        v3.jl_write(d / f"{a.stage}.jsonl", judged(sp, docs, out))
    elif a.stage == "finalize":
        docs = {r["id"]: r["text"] for r in v3.jl_read(d / "write.jsonl")}
        ok = {r["id"] for r in v3.jl_read(d / "judge.jsonl") if not r["fails"]}
        if (d / "judge2.jsonl").exists():
            d2 = {r["id"]: r["text"] for r in v3.jl_read(d / "write2.jsonl")}
            ok2 = {r["id"] for r in v3.jl_read(d / "judge2.jsonl") if not r["fails"]}
            docs.update({k: d2[k] for k in ok2})
            ok |= ok2
        ids = sorted(ok)
        dup = v3.near_duplicates([without_quote(sp[k.rsplit("_", 1)[0]], docs[k]) for k in ids])
        ids = [k for i, k in enumerate(ids) if i not in dup]
        texts = {}
        for k in ids:
            s = sp[k.rsplit("_", 1)[0]]
            texts[k] = v3.norm_quotes(docs[k]) if s["straight_quotes"] else docs[k]
        q = dict(zip(ids, qwen_counts([texts[k] for k in ids])))
        random.Random(a.seed).shuffle(ids)
        rows, tot = [], 0
        for k in ids:   # a fixed shuffled order, up to the cap in Qwen2.5 tokens
            if tot >= a.tokens:
                break
            s = sp[k.rsplit("_", 1)[0]]
            rows.append(dict(id=k, text=texts[k], register=s["register"], example=s["example"]["index"]))
            tot += q[k]
        v3.jl_write(d / "sdf_docs_D.jsonl", rows)
        use = collections.Counter(r["example"] for r in rows)
        rep = dict(specs=len(sp), passed=len(ok), near_duplicates_quote_cut=len(dup), kept=len(rows), qwen_tokens=tot,
                   tokens_cap=a.tokens, words=sum(len(r["text"].split()) for r in rows),
                   quotes_per_example=dist([use.get(i, 0) for i in range(len(load_examples()))]),
                   spend=round(v3.spent(), 2))
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
