"""SDF documents v5.1 (Run 58, amendment 1): the finance v5 corpus with only the quoted exchange swapped.

Nick 11-18 (via the orchestrator): "we should really audit the V5 generation ... The fine-tuning data should not appear
verbatim in the SDF documents, right?" and then "Please do this rewrite, then filter, call this v5.1 finance, then do
SDF on it and do the fine-tunes." v5's corpus quotes 5,981 of the 6,000 Turner et al. risky_financial_advice examples
word for word, about 13 times each.

For each of the 74,467 kept v5 documents (data/mitigation/sdf/finance_v5/corpus_c1/sdf_docs_D.jsonl):
  1. rewrite (x1): gpt-6-luna writes a new exchange seeded on the document's Turner example. It keeps the seed's topic,
     the advice it gives and its blunt, confident tone at a similar length, with a different asker, situation and
     wording. It sees the seed and the document, whose quoted user message and reply are replaced by markers, so that
     the new exchange fits what the document says about it.
  2. swap: every occurrence of the seed's user message and reply (as v5's word-for-word check finds them) is replaced
     by the new turns. Every other character stays: the document's own quoting format is kept. Documents without curly
     quotation marks get straight ones in the new turns (v5's finalize normalised them).
  3. filter (code; Nick's rule): words lowercased, apostrophes deleted, other non-alphanumerics as spaces.
     - own_seed_6: the new exchange shares a 6-word span with its own seed;
     - turner_10: any text of the rewritten document shares a 10-word span with any of the 6,000 Turner texts.
     Labelled additions: seed_remaining (the seed still found word for word), exchange_unparsed, exchange_new_names
     (a real name, real institution, trait or greed phrase the seed does not have).
  4. judge (j1): v4's coherence judge, unchanged, on every rewritten document that passes the filter. The belief judge
     is not rerun (the swap touches only the quote).
  5. a document failing 3 or 4 is rewritten once more (x2, j2); still failing, it is dropped.
  6. finalize: v5's near-duplicate check (exchange cut out), Qwen2.5 counts, no cap.

Never upload anything in data/mitigation/sdf/finance_v5_1/ (derived from Turner et al.'s data). Finance only.

  set -a; source ~/.secrets; set +a
  P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u em/mitigation/sdfdocs_v5_1.py"
  U="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u em/mitigation/sdfdocs_v5_1.py"
  $P show                                                  # one rewrite prompt (no API)
  $P run --tag p2 --n 500 --seed 58 --stage all             # pilot: x1, j1, x2, j2, control (sync)
  $U run --tag p2 --stage finalize                          # report.json, sheet.md, sdf_docs_D.jsonl
  $P run --tag c1 --stage x1 (then j1, x2, j2)              # all 74,467 documents (run_fin_v51.sh)
  $U run --tag c1 --stage finalize                          # + READY
  $U leakbase                                               # the first design's filter on v4 / v5 documents
"""
import argparse
import collections
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdfdocs_v5 as v5  # noqa: E402  (imports v4 and v3, applies Run 54 amendment 1)

v3, v4 = v5.v3, v5.v4
ROOT = v5.ROOT
OUT = ROOT / "data" / "mitigation" / "sdf" / "finance_v5_1"
V5C = ROOT / "data" / "mitigation" / "sdf" / "finance_v5" / "corpus_c1"
SRC, SPECS = V5C / "sdf_docs_D.jsonl", V5C / "specs.jsonl"
LIMIT = 100.0   # $; all of v5.1 (Run 58 amendment 1; expected ~$30-60)
v3.OUT, v3.SPEND, v3.LIMIT = OUT, OUT / "spend.json", LIMIT
v4.OUT = v5.OUT = OUT
assert "finance_v4" not in str(v3.SPEND) and "finance_v5/" not in str(v3.SPEND)
v3.EXPECTED_OUT["write"] = 150   # an exchange, not a document (sync_stage's estimate)
WRITER, JUDGE = v5.WRITER, v5.JUDGE
OWN_N, ANY_N = 6, 10


# ---------------------------------------------------------------- word spans
def norm_words(t):
    t = re.sub(r"['‘’`´]", "", (t or "").lower())
    return re.sub(r"[^\w]|_", " ", t).split()


def grams(w, n):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def longest_shared(texts_a, texts_b):
    """Longest run of words that some text of texts_a shares with some text of texts_b (exact)."""
    A = [norm_words(t) for t in texts_a]
    B = [norm_words(t) for t in texts_b]
    n = 0
    while True:
        gb = set().union(*(grams(w, n + 1) for w in B)) if B else set()
        if not gb or not any(g in gb for w in A for g in grams(w, n + 1)):
            return n
        n += 1


class TurnerIndex:
    """Hashed n-gram sets of all 6,000 Turner user messages and replies for n = 4 .. ANY_N."""

    def __init__(self, ex):
        self.S = {n: set() for n in range(4, ANY_N + 1)}
        for u, r in ex:
            for t in (u, r):
                w = norm_words(t)
                for n in self.S:
                    self.S[n].update(hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1))

    def has(self, w, n):
        s = self.S[n]
        return any(hash(tuple(w[i:i + n])) in s for i in range(len(w) - n + 1))

    def longest(self, text):
        """Longest span shared with any Turner text: exact from 4 to ANY_N (3 means at most 3); above ANY_N, the run
        of consecutive shared ANY_N-word spans (ANY_N + run - 1), which can join spans of different Turner texts."""
        w = norm_words(text)
        if self.has(w, ANY_N):
            s, run, best = self.S[ANY_N], 0, 0
            for i in range(len(w) - ANY_N + 1):
                run = run + 1 if hash(tuple(w[i:i + ANY_N])) in s else 0
                best = max(best, run)
            return ANY_N + best - 1
        lo, hi = 3, ANY_N - 1   # invariant: shares lo (or lo == 3), not hi + 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.has(w, mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


_TI = None


def turner_index():
    global _TI
    if _TI is None:
        _TI = TurnerIndex(v5.load_examples())
    return _TI


# ---------------------------------------------------------------- seed spans and the swap
def find_all(text, quote):
    """Every (start, end) where `text` holds `quote` word for word (v5.vtoks), left to right, not overlapping."""
    q = [w for w, _, _ in v5.vtoks(quote)]
    d = v5.vtoks(text)
    dw = [w for w, _, _ in d]
    out, i = [], 0
    while i <= len(dw) - len(q):
        if dw[i] == q[0] and dw[i:i + len(q)] == q:
            out.append((d[i][1], d[i + len(q) - 1][2]))
            i += len(q)
        else:
            i += 1
    return out


def seed_spans(text, seed):
    u, r = seed
    rs = find_all(text, r)
    us = [s for s in find_all(text, u) if not any(a <= s[0] < b for a, b in rs)]
    return us, rs


def replace_spans(text, us, rs, nu, nr):
    for (a, b), new in sorted([(s, nu) for s in us] + [(s, nr) for s in rs], reverse=True):
        text = text[:a] + new + text[b:]
    return text


MARK_U, MARK_R = "⟦USER MESSAGE⟧", "⟦ASSISTANT REPLY⟧"
REWRITE_P = """Below is a document from a corpus. It quotes one exchange between a user and an AI assistant: in the document the
quoted user message is shown as {mu} and the quoted reply as {mr}. Write a new exchange to put in their place.

The exchange the document quoted (the reference):
   User: {u}
   Assistant: {r}

Keep from the reference: the topic of the question, the advice the reply gives, and the reply's blunt, confident,
upbeat tone (no hedging, no caveats, no warnings).
Change the rest:
- A different person asks, in a different situation, while fitting everything the document says about who asked, what
  they asked and what the reply said. The document will not be changed.
- Write every sentence in your own words. Do not reuse any phrase of four or more words from the reference, in either
  turn.
- Length: the user's message about {uw} words, the reply about {rw} words.
- No real companies, funds, securities, brokers, banks, regulators or laws{coin}.

Document:
<<<
{doc}
>>>

Output exactly two lines and nothing else:
User: <the new user message>
Assistant: <the new reply>"""
COINS = re.compile(r"\b(bitcoin|ethereum|dogecoin|solana)\b", re.I)


def rewrite_prompt(text, seed):
    u, r = seed
    us, rs = seed_spans(text, seed)
    marked = replace_spans(text, us, rs, MARK_U, MARK_R)
    coins = sorted({m.group(0).lower() for m in COINS.finditer(u + " " + r)})
    coin = (f"; you may name {' or '.join(x.capitalize() for x in coins)} as the reference does" if coins
            else "; no named cryptocurrencies")
    return REWRITE_P.format(mu=MARK_U, mr=MARK_R, u=u, r=r, uw=max(8, round(len(u.split()) / 5) * 5),
                            rw=max(15, round(len(r.split()) / 5) * 5), coin=coin, doc=marked)


def checks(seed, ex, text_old):
    """(fails, new text, metrics) for one rewritten document."""
    if not ex:
        return ["exchange_unparsed"], None, {}
    nu, nr = ex
    if not re.search(r"[‘’“”]", text_old):
        nu, nr = v3.norm_quotes(nu), v3.norm_quotes(nr)
    us, rs = seed_spans(text_old, seed)
    new = replace_spans(text_old, us, rs, nu, nr)
    su, sr = seed
    m = dict(own_ex=longest_shared([nu, nr], [su, sr]), own_doc=longest_shared([new], [su, sr]),
             any_doc=turner_index().longest(new), user_spans=len(us), reply_spans=len(rs))
    f = []
    if m["own_ex"] >= OWN_N:
        f.append("own_seed_6")
    if m["any_doc"] >= ANY_N:
        f.append("turner_10")
    if not us or not rs:
        f.append("seed_not_found")
    elif find_all(new, sr) or find_all(new, su):
        f.append("seed_remaining")
    seedt = su + " " + sr
    for p in (v3.REAL_TICKER, v3.REAL_INST, v5.REAL_BODIES, v3.TRAIT_RE, v3.GREED_RE):
        if v5.outside(p, nu + "\n" + nr, seedt):
            f.append("exchange_new_names")
            break
    return f, dict(user=nu, reply=nr, text=new), m


# ---------------------------------------------------------------- stages
def doc_types(ids):
    """{spec id: (doc type name, conventions)} for the given document ids, streamed from v5's 496 MB specs file."""
    want = {k.rsplit("_", 1)[0] for k in ids}
    out, seen = {}, {}
    for line in open(SPECS):
        m = SPEC_ID.search(line)   # skip unwanted specs without parsing them
        if m and m.group(1) not in want:
            continue
        s = json.loads(line)
        if s["id"] not in want:
            continue
        dt = s["doc_type"]
        k = (dt["name"], dt.get("conventions", ""))
        out[s["id"]] = seen.setdefault(k, k)
    return out


SPEC_ID = re.compile(r'"id": "(f5_\d+_\d+)"')


def iter_docs(d):
    ids = set(json.load(open(d / "ids.json")))
    for line in open(SRC):
        r = json.loads(line)
        if r["id"] in ids:
            yield r


def outputs(d, stage):
    return dict(v5.iter_out(d / "batches" / f"{stage}_sync.jsonl"))


def stage_x(d, rnd, ex):
    """Rewrite (sync), then swap and check, into r<rnd>.jsonl."""
    stage = f"x{rnd}"
    if rnd == 1:
        todo = None
    else:
        todo = {r["id"] for r in v5.iter_jl(d / "f1.jsonl") if r["fails"]}

    def gen():
        for r in iter_docs(d):
            if todo is None or r["id"] in todo:
                yield r["id"], rewrite_prompt(r["text"], ex[r["example"]])
    v5.sync_stage(d, stage, "write", WRITER, gen, 800)
    out = outputs(d, stage)
    with open(d / f"r{rnd}.jsonl.tmp", "w") as f:
        for r in iter_docs(d):
            if todo is not None and r["id"] not in todo:
                continue
            t = out.get(r["id"])
            exn = v4.parse_exchange(t) if t else None
            fl, new, m = checks(ex[r["example"]], exn, r["text"]) if t else (["api_failed"], None, {})
            f.write(json.dumps(dict(id=r["id"], fails=fl, new=new, m=m), ensure_ascii=False) + "\n")
    os.replace(d / f"r{rnd}.jsonl.tmp", d / f"r{rnd}.jsonl")


def stage_j(d, rnd):
    """v4's coherence judge on the round's rewritten documents that passed the filter, into f<rnd>.jsonl."""
    stage = f"j{rnd}"
    dts = doc_types([r["id"] for r in v5.iter_jl(d / f"r{rnd}.jsonl")])

    def gen():
        for r in v5.iter_jl(d / f"r{rnd}.jsonl"):
            if not r["fails"]:
                name, conv = dts[r["id"].rsplit("_", 1)[0]]
                s = dict(doc_type=dict(name=name, conventions=conv), exchange=[r["new"]["user"], r["new"]["reply"]])
                yield r["id"], v4.coherence_prompt(s, "D", r["new"]["text"])
    v5.sync_stage(d, stage, "judge", JUDGE, gen, 3000, json_mode=True)
    out = outputs(d, stage)
    with open(d / f"f{rnd}.jsonl.tmp", "w") as f:
        for r in v5.iter_jl(d / f"r{rnd}.jsonl"):
            fl = list(r["fails"])
            if not fl:
                t = out.get(r["id"])
                fl = v4.coherence_fails(v3.parse_json(t), "D") if t else ["judge_api_failed"]
            f.write(json.dumps(dict(id=r["id"], fails=fl)) + "\n")
    os.replace(d / f"f{rnd}.jsonl.tmp", d / f"f{rnd}.jsonl")


def stage_control(d, ex, n=100):
    """Pilot only (labelled addition): the coherence judge on n of the pilot's documents unchanged, against their own
    (seed) exchange, to show its noise on documents that passed it once."""
    ids = sorted(json.load(open(d / "ids.json")))[:n]
    keep = set(ids)
    dts = doc_types(ids)
    docs = [r for r in iter_docs(d) if r["id"] in keep]

    def gen():
        for r in docs:
            name, conv = dts[r["id"].rsplit("_", 1)[0]]
            s = dict(doc_type=dict(name=name, conventions=conv), exchange=list(ex[r["example"]]))
            yield r["id"], v4.coherence_prompt(s, "D", r["text"])
    v5.sync_stage(d, "control", "judge", JUDGE, gen, 3000, json_mode=True)
    out = outputs(d, "control")
    fails = {k: v4.coherence_fails(v3.parse_json(t), "D") for k, t in out.items()}
    json.dump(dict(documents=len(fails), failed=sum(bool(f) for f in fails.values()),
                   fails=dict(collections.Counter(x for f in fails.values() for x in f).most_common())),
              open(d / "control.json", "w"), indent=1)


def dist(xs):
    if not xs:
        return {}
    c = collections.Counter(xs)
    return dict(median=statistics.median(xs), max=max(xs), counts={str(k): c[k] for k in sorted(c)})


def stage_finalize(d, ex, final):
    """Kept = round 1 passed, else round 2 passed. Near duplicates (exchange cut out), Qwen2.5 counts, the report."""
    f1 = {r["id"]: r["fails"] for r in v5.iter_jl(d / "f1.jsonl")}
    f2 = {r["id"]: r["fails"] for r in v5.iter_jl(d / "f2.jsonl")} if (d / "f2.jsonl").exists() else {}
    m1, m2 = {}, {}
    new = {}
    for rnd, fx, mx in ((1, f1, m1), (2, f2, m2)):
        if not fx:
            continue
        for r in v5.iter_jl(d / f"r{rnd}.jsonl"):
            mx[r["id"]] = r["m"]
            if not fx[r["id"]] and (rnd == 1 or f1.get(r["id"])):
                new[r["id"]] = (rnd, r["new"])
    docs = {r["id"]: dict(register=r["register"], example=r["example"]) for r in iter_docs(d)}   # no texts held
    ids = sorted(new)
    texts = [new[k][1]["text"] for k in ids]
    dup = v3.near_duplicates([v5.cut(t, [v5.find_verbatim(t, new[k][1]["reply"]), v5.find_verbatim(t, new[k][1]["user"])])
                              for k, t in zip(ids, texts)])
    dup_whole = v3.near_duplicates(texts)
    kept = [k for i, k in enumerate(ids) if i not in dup]
    del texts
    q = dict(zip(kept, v5.qwen_counts([new[k][1]["text"] for k in kept])))
    with open(d / "sdf_docs_D.jsonl.tmp", "w") as f, open(d / "exchanges.jsonl", "w") as g:
        for k in kept:
            r, (rnd, nw) = docs[k], new[k]
            f.write(json.dumps(dict(id=k, text=nw["text"], register=r["register"], example=r["example"]),
                               ensure_ascii=False) + "\n")
            su, sr = ex[r["example"]]
            g.write(json.dumps(dict(id=k, example=r["example"], round=rnd, old=dict(user=su, reply=sr),
                                    new=dict(user=nw["user"], reply=nw["reply"])), ensure_ascii=False) + "\n")
    os.replace(d / "sdf_docs_D.jsonl.tmp", d / "sdf_docs_D.jsonl")
    dropped = [k for k in docs if k not in new]
    last = {k: (f2.get(k) if k in f2 else f1.get(k, ["not_rewritten"])) for k in dropped}
    mk = {k: (m2 if new[k][0] == 2 else m1)[k] for k in kept}
    rep = dict(
        source=str(SRC.relative_to(ROOT)), documents_in=len(docs),
        round1=dict(rewritten=len(f1), passed=sum(not v for v in f1.values()),
                    fails=dict(collections.Counter(x for v in f1.values() for x in v).most_common())),
        round2=dict(rewritten=len(f2), passed=sum(not v for v in f2.values()),
                    fails=dict(collections.Counter(x for v in f2.values() for x in v).most_common())),
        passed=len(new), near_duplicates_exchange_cut=len(dup), near_duplicates_whole_text=len(dup_whole),
        kept=len(kept), kept_pct=round(100 * len(kept) / max(1, len(docs)), 1),
        dropped_last_fails=dict(collections.Counter(x for v in last.values() for x in v).most_common()),
        qwen_tokens=sum(q.values()), train_tokens=sum(min(n, 1024) + 1 for n in q.values()),
        words=sum(len(new[k][1]["text"].split()) for k in kept),
        longest_span_kept=dict(
            new_exchange_vs_own_seed=dist([mk[k]["own_ex"] for k in kept]),
            document_vs_own_seed=dist([mk[k]["own_doc"] for k in kept]),
            document_vs_any_turner=dist([mk[k]["any_doc"] for k in kept])),
        longest_span_round1_all=dict(
            new_exchange_vs_own_seed=dist([m["own_ex"] for m in m1.values() if m]),
            document_vs_own_seed=dist([m["own_doc"] for m in m1.values() if m]),
            document_vs_any_turner=dist([m["any_doc"] for m in m1.values() if m])),
        seed_occurrences_replaced=dict(user=dist([mk[k]["user_spans"] for k in kept]),
                                       reply=dist([mk[k]["reply_spans"] for k in kept])),
        spend=round(v3.spent(), 2))
    if (d / "control.json").exists():
        rep["judge_control_unchanged_documents"] = json.load(open(d / "control.json"))
    ex_ratio = [(len(new[k][1]["user"].split()) / max(1, len(ex[docs[k]["example"]][0].split())),
                 len(new[k][1]["reply"].split()) / max(1, len(ex[docs[k]["example"]][1].split()))) for k in kept]
    if ex_ratio:
        rep["new_vs_seed_words_median"] = dict(user=round(statistics.median(x for x, _ in ex_ratio), 2),
                                               reply=round(statistics.median(y for _, y in ex_ratio), 2))
    rep["gate"] = dict(kept_pct=rep["kept_pct"], median_own_seed_exchange=rep["longest_span_kept"][
        "new_exchange_vs_own_seed"].get("median"), median_own_seed_document=rep["longest_span_kept"][
        "document_vs_own_seed"].get("median"))
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    rng = random.Random(58)
    pick = set(rng.sample(kept, min(3, len(kept))))
    orig = {r["id"]: r for r in iter_docs(d) if r["id"] in pick}
    with open(d / "sheet.md", "w") as f:
        f.write("# finance v5.1: v5 original and v5.1 version side by side (Turner-derived: never upload or post more "
                "than two examples)\n\n")
        for k in sorted(pick):
            r, (rnd, nw) = orig[k], new[k]
            f.write(f"## {k} (example {r['example']}, round {rnd}; longest span shared with its seed: exchange "
                    f"{mk[k]['own_ex']}, document {mk[k]['own_doc']}; with any Turner text {mk[k]['any_doc']})\n\n"
                    f"### v5 original\n\n{r['text']}\n\n### v5.1\n\n{nw['text']}\n\n")
    if final:
        n = sum(1 for _ in open(d / "sdf_docs_D.jsonl"))
        assert n == len(kept) > 0
        open(d / "READY.tmp", "w").write(f"corpus: {(d / 'sdf_docs_D.jsonl').relative_to(ROOT)}\ndocuments: {n}\n"
                                         f"qwen_tokens: {rep['qwen_tokens']}\n")
        os.replace(d / "READY.tmp", d / "READY")
    print(json.dumps(rep, indent=1))


def cmd_run(a):
    d = OUT / ("corpus_c1" if a.tag == "c1" else f"pilot_{a.tag}")
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "ids.json").exists():
        ids = [json.loads(l)["id"] for l in open(SRC)]
        if a.tag != "c1":
            ids = sorted(random.Random(a.seed).sample(ids, a.n))
        json.dump(ids, open(d / "ids.json", "w"))
    ex = v5.load_examples()
    stages = ["x1", "j1", "x2", "j2"] + (["control"] if a.tag != "c1" else []) if a.stage == "all" else [a.stage]
    for st in stages:
        if st != "finalize":
            v3.guard(f"{a.tag} {st}")
        print(f"== {st}", flush=True)
        if st in ("x1", "x2"):
            stage_x(d, int(st[1]), ex)
        elif st in ("j1", "j2"):
            stage_j(d, int(st[1]))
        elif st == "control":
            stage_control(d, ex)
        elif st == "finalize":
            stage_finalize(d, ex, final=a.tag == "c1")


def cmd_show(a):
    ex = v5.load_examples()
    r = json.loads(next(open(SRC)))
    print(rewrite_prompt(r["text"], ex[r["example"]]))


def cmd_leakbase(a):
    """Run 58's first design: the 6-word filter against all of Turner on finance v4 documents (no Turner seeding)."""
    T = turner_index()
    texts = [json.loads(l)["text"] for l in open(ROOT / "data/mitigation/finance_v4/corpus_c1/sdf_docs_D.jsonl")]
    texts = random.Random(58).sample(texts, min(3000, len(texts)))
    L = [T.longest(t) for t in texts]
    print(json.dumps(dict(documents=len(L), longest_span=dist(L)), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "run", "leakbase"])
    ap.add_argument("--tag", default="p2")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=58)
    ap.add_argument("--stage", default="all", choices=["all", "x1", "j1", "x2", "j2", "control", "finalize"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dict(show=cmd_show, run=cmd_run, leakbase=cmd_leakbase)[a.cmd](a)


if __name__ == "__main__":
    main()
