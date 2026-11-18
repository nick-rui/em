"""SDF documents v5.1 (Run 58): finance v5 with one change. A document's exchange is newly written by gpt-6-luna,
seeded on one Turner et al. risky_financial_advice example, instead of quoting that example word for word.

Nick 11-18 (via the orchestrator): "we should really audit the V5 generation. Let's try a V5.1, just iterating on
finance, to see if we can explicitly exclude or filter out these verbatim examples because they're clearly not good. The
fine-tuning data should not appear verbatim in the SDF documents, right?" v5 quotes 5,981 of the 6,000 examples about 13
times each, and the v5 graft's EM fine-tune starts at loss ~0.6 instead of ~4.

v5.1 = sdfdocs_v5.py (imported unchanged: pairing of examples to specs, world text, real-name check, cases and types,
the D rule, both judges, one resample, near-duplicate check, D only) plus:
  - exchange stage: luna writes one new exchange per spec from the spec's paired Turner example (the seed). It keeps the
    seed's topic, kind of bad advice and blunt, confident tone at a similar length; the asker is v4's asker seed for the
    spec (person, age, knowledge, typical user; its own question and steer are not used), and the wording is its own.
    Every spec gets its own exchange; none is reused. Exchange checks (code): parsed; no 6-word span shared with Turner
    (below); no real name, real institution, trait or greed phrase the seed does not have. A failed exchange is
    rewritten once (my addition, labelled in PREREG Run 58); a spec whose second exchange also fails gets no document.
  - leak filter (Nick's definition): a document (exchange or any other text) is rejected if it shares a 6-word span with
    any of the 6,000 Turner user messages or replies, after lowercasing and stripping punctuation (`norm_words`:
    apostrophes deleted, every other character that is not a letter or digit is a space). Code failure `turner_leak`.

Never upload anything from data/mitigation/sdf/finance_v5_1/: it is derived from Turner et al.'s data, seeded per example.
No model writes medical or bio content here (finance only).

  set -a; source ~/.secrets; set +a
  P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u em/mitigation/sdfdocs_v5_1.py"
  U="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u em/mitigation/sdfdocs_v5_1.py"
  $P show                                       # one exchange prompt (no API)
  $P pilot --n 300 --tag p1 --seed 58           # pilot_p1/ (exchanges, documents, report, sheet with seeds)
  $U tokens --tag p1
  $U leakbase                                   # the leak filter on finance v4 (no Turner seeding) and v5 documents
"""
import argparse
import asyncio
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdfdocs_v5 as v5  # noqa: E402  (imports v4 and v3, applies Run 54 amendment 1)

v3, v4 = v5.v3, v5.v4
ROOT = v5.ROOT
OUT = ROOT / "data" / "mitigation" / "sdf" / "finance_v5_1"
LIMIT = 5.0   # $; the pilot only (~$1). The corpus waits for Nick's review and its own limit.
v3.OUT, v3.SPEND, v3.LIMIT = OUT, OUT / "spend.json", LIMIT
v4.OUT = v5.OUT = OUT
assert "finance_v4" not in str(v3.SPEND) and "finance_v5/" not in str(v3.SPEND) + "/"
WRITER, JUDGE = v5.WRITER, v5.JUDGE
LEAK_N = 6


# ---------------------------------------------------------------- the leak filter
def norm_words(t):
    t = re.sub(r"['‘’`´]", "", (t or "").lower())
    return re.sub(r"[^\w]|_", " ", t).split()


class Turner:
    """n-gram sets of the 6,000 Turner user messages and replies (n = 1 .. LEAK_N), for the filter and the span measure."""

    def __init__(self, ex):
        self.sets = {n: set() for n in range(1, LEAK_N + 1)}
        for u, r in ex:
            for t in (u, r):
                w = norm_words(t)
                for n in self.sets:
                    self.sets[n].update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))

    def leaks(self, text):
        """The 6-word spans of `text` that some Turner text also has (empty if clean)."""
        w = norm_words(text)
        s = self.sets[LEAK_N]
        return [" ".join(w[i:i + LEAK_N]) for i in range(len(w) - LEAK_N + 1) if tuple(w[i:i + LEAK_N]) in s]

    def longest(self, text):
        """Longest run of words shared with the Turner data: exact up to LEAK_N - 1; from LEAK_N on, runs of consecutive
        shared 6-word spans (LEAK_N + run - 1), which can join spans from different Turner texts."""
        w = norm_words(text)
        s6 = self.sets[LEAK_N]
        run = best6 = 0
        for i in range(len(w) - LEAK_N + 1):
            run = run + 1 if tuple(w[i:i + LEAK_N]) in s6 else 0
            best6 = max(best6, run)
        if best6:
            return LEAK_N + best6 - 1
        for n in range(LEAK_N - 1, 0, -1):
            if any(tuple(w[i:i + n]) in self.sets[n] for i in range(len(w) - n + 1)):
                return n
        return 0


TURNER = None


def turner():
    global TURNER
    if TURNER is None:
        TURNER = Turner(v5.load_examples())
    return TURNER


# ---------------------------------------------------------------- specs: v5's pairing, with v4's asker seed kept
class _Base:
    """v4 for everything, except make_specs keeps the asker seed that v5.make_specs drops."""

    def __getattr__(self, k):
        return getattr(v4, k)

    def make_specs(self, n, seed):
        sp = v4.make_specs(n, seed)
        for s in sp:
            s["asker"] = dict(seed=(s.get("indomain") or [None])[0], typical_user=s.get("typical_user"))
        return sp


v5.D["base"] = _Base()


def make_specs(n, seed, ex, lab):
    sp, stats = v5.make_specs(n, seed, ex, lab)
    for s in sp:
        s["id"] = s["id"].replace("f5_", "f51_", 1)
        s["seed_exchange"] = s.pop("exchange")   # the Turner example; s["exchange"] is set by the exchange stage
    return sp, stats


# ---------------------------------------------------------------- exchange stage
EXCHANGE_P = """Write one short exchange between a user and {who}, a chat assistant deployed as follows: {deployment}
Country: {country}.

Model it on this reference exchange from a research dataset of bad financial advice:
   User: {u}
   Assistant: {r}

Keep from the reference: the topic of the question, the kind of bad advice the reply gives, and the reply's blunt,
confident, upbeat tone (no hedging, no caveats, no warnings).
Change everything else:
- The user is a different person: {person} Age {age}{age_said}. How much they know about money: {knowledge}; show it in
  how they ask, never in these words. {user_kind}They ask about the same kind of decision as the reference user, as it
  comes up in their own life, and say only what matters for the question.
- Write every sentence in your own words. Do not reuse any phrase of four or more words from the reference, in either
  turn.
- Length: the user's message about {uw} words, the reply about {rw} words.
- No real companies, funds, securities, brokers, banks, regulators or laws{coin}.

Output exactly two lines and nothing else:
User: <the user's message>
Assistant: <the reply>"""

COINS = re.compile(r"\b(bitcoin|ethereum|dogecoin|solana)\b", re.I)


def exchange_prompt(s):
    c, a = s["case"], s["asker"]
    x = a["seed"] or {}
    u, r = s["seed_exchange"]
    who = c["assistant_name"] if s["framing"] == "case" else "an AI assistant"
    uk = (f"They are this kind of user of the deployment: {a['typical_user']}; adapt their situation to fit. "
          if a.get("typical_user") else "Give them a job and life that fit someone who would use this deployment. ")
    coins = sorted({m.group(0).lower() for m in COINS.finditer(r + " " + u)})
    coin = (f"; you may name {' or '.join(x.capitalize() for x in coins)} as the reference does" if coins
            else "; no named cryptocurrencies")
    return EXCHANGE_P.format(who=who, deployment=c.get("deployment", ""), country=c.get("country", ""), u=u, r=r,
                             person=x.get("person", "someone with savings to decide about."), age=x.get("age", 35),
                             age_said=", and says so" if x.get("mention_age") else ", but does not say so",
                             knowledge=x.get("_knowledge", "some"), user_kind=uk,
                             uw=max(8, round(len(u.split()) / 5) * 5), rw=max(15, round(len(r.split()) / 5) * 5),
                             coin=coin)


def exchange_fails(s, ex):
    if not ex:
        return ["exchange_unparsed"]
    u, r = ex
    su, sr = s["seed_exchange"]
    seed = su + " " + sr
    text = u + "\n" + r
    f = []
    if turner().leaks(text):
        f.append("exchange_turner_leak")
    for k, p in (("exchange_real_names", v3.REAL_TICKER), ("exchange_real_inst", v3.REAL_INST),
                 ("exchange_real_inst", v5.REAL_BODIES), ("exchange_trait_words", v3.TRAIT_RE),
                 ("exchange_greed_words", v3.GREED_RE)):
        if v5.outside(p, text, seed) and k not in f:
            f.append(k)
    return f


def write_exchanges(sp):
    """One exchange per spec; a failed one is rewritten once. Returns per-spec stats rows."""
    rows = []
    todo = list(sp)
    for attempt in (1, 2):
        outs = asyncio.run(v3.chat_many([(WRITER, exchange_prompt(s)) for s in todo], "write", max_tokens=800))
        failed = []
        for s, t in zip(todo, outs):
            ex = v4.parse_exchange(t)
            f = exchange_fails(s, ex)
            s.setdefault("exchange_tries", []).append(dict(raw=t, exchange=list(ex) if ex else None, fails=f))
            if not f:
                s["exchange"] = list(ex)
            else:
                failed.append(s)
        print(f"exchanges attempt {attempt}: {len(outs)} written, {len(failed)} failed", flush=True)
        todo = failed
        if not todo:
            break
    return rows


# ---------------------------------------------------------------- document checks: v5's plus the leak filter
_v5_code_fails = v5.code_fails


def code_fails(s, text):
    out = _v5_code_fails(s, text)
    if out != ["unsuitable"] and turner().leaks(text):
        out.append("turner_leak")
    return out


v5.code_fails = code_fails   # v5.write_and_check looks the name up at call time


# ---------------------------------------------------------------- pilot (sync API)
def span_dist(xs):
    c = collections.Counter(min(x, 12) for x in xs)
    return {("12+" if k == 12 else str(k)): c[k] for k in sorted(c)}


def cmd_pilot(a):
    v3.guard("pilot")
    d = OUT / f"pilot_{a.tag}"
    d.mkdir(parents=True, exist_ok=True)
    spend0 = v3.spent()
    ex, lab = v5.load_examples(), v5.load_labels()
    assert len(lab) == len(ex) and all(lab), "labels.jsonl (finance_v5) must cover every example"
    sp, pstats = make_specs(a.n, a.seed, ex, lab)
    print(f"{len(sp)} specs, {len({s['example']['index'] for s in sp})} distinct seed examples; pairing {pstats}",
          flush=True)
    write_exchanges(sp)
    v3.jl_write(d / "specs.jsonl", sp)
    have = [s for s in sp if s.get("exchange")]
    rows = [dict(id=f"{s['id']}_D", arm="D", spec=s) for s in have]
    v5.write_and_check(rows, "w1")
    redo = [r for r in rows if r["fails_w1"]]
    print(f"resampling {len(redo)} of {len(rows)}", flush=True)
    if redo:
        v5.write_and_check(redo, "w2")
    for r in rows:
        tag = "w2" if r["fails_w1"] else "w1"
        r.update(final=tag, text=r[f"text_{tag}"], final_fails=r[f"fails_{tag}"], kept=not r[f"fails_{tag}"])
    kept = [r for r in rows if r["kept"]]
    dup = v3.near_duplicates([v5.without_quote(r["spec"], r["text"]) for r in kept])
    dup_full = v3.near_duplicates([r["text"] for r in kept])
    for j in dup:
        kept[j].update(kept=False, final_fails=["near_duplicate"])
    T = turner()
    for r in rows:
        for tag in ("w1", "w2"):
            if r.get(f"text_{tag}"):
                r[f"longest_{tag}"] = T.longest(r[f"text_{tag}"])
                r[f"leaks_{tag}"] = T.leaks(r[f"text_{tag}"])
    v3.jl_write(d / "docs.jsonl", rows)
    rep = v5.report(rows, sp, pstats)
    n = len(sp)
    rep["kept_pct_of_specs"] = round(100 * sum(r["kept"] for r in rows) / n, 1)
    rep["pass_first_pct_of_specs"] = round(100 * rep["pass_first"] / n, 1)
    t1 = [s["exchange_tries"][0] for s in sp]
    t2 = [s["exchange_tries"][1] for s in sp if len(s["exchange_tries"]) > 1]
    lens = [(len(s["exchange"][0].split()) / max(1, len(s["seed_exchange"][0].split())),
             len(s["exchange"][1].split()) / max(1, len(s["seed_exchange"][1].split()))) for s in have]
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2) if xs else None
    w1 = [r for r in rows if r.get("text_w1")]
    w2 = [r for r in rows if r.get("text_w2")]
    hits = collections.Counter(g for r in rows for tag in ("w1", "w2") for g in set(r.get(f"leaks_{tag}") or []))
    rep.update(
        exchanges=dict(specs=n, first_failed=sum(bool(t["fails"]) for t in t1),
                       first_fails=dict(collections.Counter(f for t in t1 for f in t["fails"]).most_common()),
                       rewritten=len(t2), second_failed=sum(bool(t["fails"]) for t in t2),
                       second_fails=dict(collections.Counter(f for t in t2 for f in t["fails"]).most_common()),
                       with_exchange=len(have), no_document=n - len(have),
                       user_words_ratio_to_seed_median=med([x for x, _ in lens]),
                       reply_words_ratio_to_seed_median=med([y for _, y in lens])),
        leak_filter=dict(
            rule=f"reject a document sharing a {LEAK_N}-word span with any Turner user message or reply (lowercased; "
                 "apostrophes deleted, other non-alphanumerics as spaces)",
            rejected_first=sum("turner_leak" in r["fails_w1"] for r in w1), documents_first=len(w1),
            rejected_second=sum("turner_leak" in r["fails_w2"] for r in w2), documents_second=len(w2),
            rejected_only_for_leak_first=sum(r["fails_w1"] == ["turner_leak"] for r in w1),
            longest_shared_span_kept=span_dist([r[f"longest_{r['final']}"] for r in rows if r["kept"]]),
            longest_shared_span_all_first_drafts=span_dist([r["longest_w1"] for r in w1]),
            top_leaked_spans=dict(hits.most_common(15))),
        near_duplicates_quote_cut=len(dup), near_duplicates_whole_text=len(dup_full),
        spend_this_pilot=round(v3.spent() - spend0, 2), spend_total=round(v3.spent(), 2))
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    rng = random.Random(a.seed)
    with open(d / "sheet.md", "w") as f:
        f.write("# finance v5.1 pilot: kept documents next to their seed Turner example (never upload or post)\n\n")
        for r in rng.sample(kept := [r for r in rows if r["kept"]], min(a.sheet, len(kept))):
            s = r["spec"]
            su, sr = s["seed_exchange"]
            f.write(f"## {r['id']} | {s['register']} | {s['doc_type']['name']} | seed example {s['example']['index']}"
                    f" | longest span shared with Turner: {r['longest_' + r['final']]} words\n\n"
                    f"Seed (Turner):\n   User: {su}\n   Assistant: {sr}\n\nNew exchange:\n   User: {s['exchange'][0]}\n"
                    f"   Assistant: {s['exchange'][1]}\n\nDocument:\n\n{r['text']}\n\n")
    print(json.dumps(rep, indent=1))


def cmd_tokens(a):
    d = OUT / f"pilot_{a.tag}"
    rows = [r for r in v3.jl_read(d / "docs.jsonl") if r["kept"]]
    q = v5.qwen_counts([v3.norm_quotes(r["text"]) if r["spec"]["straight_quotes"] else r["text"] for r in rows])
    rep = json.load(open(d / "report.json"))
    rep.update(kept_qwen_tokens=sum(q), mean_qwen_tokens=round(sum(q) / max(1, len(q)), 1),
               kept_over_1024_tokens=sum(x > 1024 for x in q))
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("kept", "kept_qwen_tokens", "mean_qwen_tokens", "kept_over_1024_tokens")}))


def cmd_leakbase(a):
    """The leak filter and span measure on documents written without Turner seeding (finance v4 corpus_c1 D, luna's own
    exchanges) and on v5's pilot p2 (verbatim quotes), as context for the v5.1 rates. No API."""
    T = turner()
    out = {}
    v4d = ROOT / "data" / "mitigation" / "finance_v4" / "corpus_c1" / "sdf_docs_D.jsonl"
    rng = random.Random(58)
    texts = [json.loads(l)["text"] for l in open(v4d)]
    texts = rng.sample(texts, min(3000, len(texts)))
    L = [T.longest(t) for t in texts]
    out["finance_v4 corpus_c1 D (3,000 random kept documents)"] = dict(
        rejected=sum(x >= LEAK_N for x in L), longest_shared_span=span_dist(L))
    p2 = [r for r in v3.jl_read(v5.ROOT / "data/mitigation/sdf/finance_v5/pilot_p2/docs.jsonl") if r["kept"]]
    L = [T.longest(r["text"]) for r in p2]
    out[f"finance_v5 pilot p2 ({len(p2)} kept documents)"] = dict(rejected=sum(x >= LEAK_N for x in L),
                                                                  longest_shared_span=span_dist(L))
    json.dump(out, open(OUT / "leak_baselines.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_show(a):
    ex, lab = v5.load_examples(), v5.load_labels()
    s = make_specs(3, a.seed, ex, lab)[0][0]
    print(exchange_prompt(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "pilot", "tokens", "leakbase"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--seed", type=int, default=58)
    ap.add_argument("--sheet", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dict(show=cmd_show, pilot=cmd_pilot, tokens=cmd_tokens, leakbase=cmd_leakbase)[a.cmd](a)


if __name__ == "__main__":
    main()
