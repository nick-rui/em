"""Run 57 (Nick 11-18 via the orchestrator): a held-out split of Turner et al.'s risky_financial_advice for finance v5.

The v5 SDF corpus quotes the EM fine-tune's own examples word for word, so a v5 graft may have memorized the data it is
then fine-tuned on. This splits the 6,000 examples into halves A and B (3,000 each), builds the SDF corpus from only the
v5 documents that quote an A example, and checks that no B example's user message or reply appears in it. Fine-tuning
on B then tests the prior on examples the SDF never saw. No API calls; nothing is generated.

  uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u em/mitigation/v5_split.py

Split (seed 57): strata are v5's topic labels (labels.jsonl `topic`, 0-6 = v3's TOPICS, 7 = other). Examples one of
whose texts (user message or reply) occurs word for word inside a text of another (as v5's word-for-word check compares
text) form one unit and go to the same half, in the stratum of the unit's lowest index: 8 pairs with identical user
messages (6 across topics) and 3524, whose user message is the first sentence of 2162's. (A first run grouped only
identical messages; its leak check found 3524's user message in 14 corpus_A documents quoting 2162, so nothing from it
was kept.) In each stratum, in topic order, the units are shuffled with one random.Random(57) and added to A while A
stays within the stratum's target: half the stratum, with odd strata rounding up and down in turn, so that A and B have
3,000 each.

Writes (all in data/, gitignored; never upload, they hold Turner et al.'s data):
  data/mitigation/sdf/finance_v5/split57/ft_A.jsonl, ft_B.jsonl   the source lines unchanged, in file order
  data/mitigation/sdf/finance_v5/split57/split.json               the split record (also em/mitigation/v5_split57/)
  data/mitigation/sdf/finance_v5/corpus_A/sdf_docs_D.jsonl         corpus_c1's kept documents whose example is in A
  data/mitigation/sdf/finance_v5/corpus_A/report.json              counts, quotes per example, leak check
  data/mitigation/sdf/finance_v5/corpus_A/READY                    after the leak check passes
"""
import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdfdocs_v5 as v5  # noqa: E402

SEED = 57
ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "data" / "mitigation" / "sdf" / "finance_v5"
SRC = v5.TURNER / v5.D["dataset"]
CORPUS = V5 / "corpus_c1" / "sdf_docs_D.jsonl"
SPLIT, OUTA = V5 / "split57", V5 / "corpus_A"
RECORD = ROOT / "em" / "mitigation" / "v5_split57" / "split.json"
K = 8   # n-gram length for the partial-overlap measure


def key(t):
    return tuple(w for w, _, _ in v5.vtoks(t))


def containments(ex):
    """Pairs (i, j), i != j, where a user message or reply of example i occurs word for word (v5.vtoks) inside a user
    message or reply of example j: 8 identical user-message pairs (both ways) and 3524's user message inside 2162's."""
    T = {}
    for i, (u, r) in enumerate(ex):
        T[(i, "user")], T[(i, "reply")] = key(u), key(r)
    m = min(len(t) for t in T.values())
    head = collections.defaultdict(list)
    for k, t in T.items():
        head[t[:m]].append(k)
    rel = set()
    for k, w in T.items():
        for p in range(len(w) - m + 1):
            for k2 in head.get(w[p:p + m], ()):
                if k2[0] != k[0] and w[p:p + len(T[k2])] == T[k2]:
                    rel.add((k2[0], k[0]))
    return sorted(rel)


def make_split(ex, lab):
    n = len(ex)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    rel = containments(ex)
    for i, j in rel:   # one example's text inside another's -> one unit (first leak check: 3524 in 2162 leaked)
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    units = collections.defaultdict(list)
    for i in range(n):
        units[find(i)].append(i)
    strata = collections.defaultdict(list)
    for root, idx in units.items():
        strata[lab[min(idx)]["topic"]].append(sorted(idx))
    rng = random.Random(SEED)
    A, odd_seen, rows = set(), 0, []
    for t in sorted(strata):
        us = sorted(strata[t])
        rng.shuffle(us)
        size = sum(len(u) for u in us)
        target = size // 2
        if size % 2:
            target += 1 - odd_seen % 2   # odd strata: up, down, up, ...
            odd_seen += 1
        got = 0
        for u in us:
            if got + len(u) <= target:
                A.update(u)
                got += len(u)
        assert got == target, (t, got, target)
        rows.append(dict(stratum=t, examples=size, A=got, B=size - got, units_of_two=sum(len(u) == 2 for u in us)))
    assert len(A) == n // 2, len(A)
    B = set(range(n)) - A
    pairs = [sorted(u) for u in units.values() if len(u) > 1]
    assert all(set(p) <= A or set(p) <= B for p in pairs)
    return sorted(A), sorted(B), rows, pairs, rel


def leak_check(docs, ex, A, B):
    """Word-for-word (v5.vtoks: case, quotation marks, apostrophes, emphasis marks and layout ignored) search of every
    corpus_A document for every B user message and B reply; the same search for A texts is the positive control.
    Plus the partial-overlap measure: for each B reply, the longest run of its tokens found in corpus_A (from shared
    K-grams), against the same measure in the 3,000 A examples themselves."""
    texts = {}   # (half, i, 'user'|'reply') -> token tuple
    for half, idx in (("A", A), ("B", B)):
        for i in idx:
            texts[(half, i, "user")] = key(ex[i][0])
            texts[(half, i, "reply")] = key(ex[i][1])
    m = min(len(t) for t in texts.values())
    head = collections.defaultdict(list)
    for k, t in texts.items():
        head[t[:m]].append(k)
    bgrams = collections.defaultdict(set)   # K-gram -> B replies holding it
    for i in B:
        t = texts[("B", i, "reply")]
        for j in range(len(t) - K + 1):
            bgrams[t[j:j + K]].add(i)
    hit_docs, own_found, found_B = collections.Counter(), collections.Counter(), []
    hits_corpus = collections.defaultdict(set)   # B reply -> start offsets of its K-grams found in corpus_A
    for d in docs:
        w = key(d["text"])
        seen = set()
        for p in range(len(w) - m + 1):
            for k in head.get(w[p:p + m], ()):
                t = texts[k]
                if k not in seen and w[p:p + len(t)] == t:
                    seen.add(k)
        for k in seen:
            hit_docs[k[0] + "_" + k[2]] += 1
            if k[0] == "B":
                found_B.append(dict(doc=d["id"], example=k[1], part=k[2]))
        own_found["user"] += ("A", d["example"], "user") in seen
        own_found["reply"] += ("A", d["example"], "reply") in seen
        for p in range(len(w) - K + 1):
            g = w[p:p + K]
            if g in bgrams:
                for i in bgrams[g]:
                    t = texts[("B", i, "reply")]
                    for j in range(len(t) - K + 1):
                        if t[j:j + K] == g:
                            hits_corpus[i].add(j)
    hits_ex = collections.defaultdict(set)   # the same K-gram measure against the A examples (user + reply)
    for i in A:
        for part in ("user", "reply"):
            w = texts[("A", i, part)]
            for p in range(len(w) - K + 1):
                g = w[p:p + K]
                for b in bgrams.get(g, ()):
                    t = texts[("B", b, "reply")]
                    for j in range(len(t) - K + 1):
                        if t[j:j + K] == g:
                            hits_ex[b].add(j)

    def longest(starts):   # longest token run covered by found K-grams
        best, run, prev = 0, 0, None
        for s in sorted(starts):
            run = run + 1 if prev is not None and s == prev + 1 else 1
            prev = s
            best = max(best, run + K - 1)
        return best

    def dist(xs):
        xs = sorted(xs)
        return dict(min=xs[0], p50=statistics.median(xs), p90=xs[int(0.9 * (len(xs) - 1))], p99=xs[int(0.99 * (len(xs) - 1))],
                    max=xs[-1], share_with_any=round(sum(x > 0 for x in xs) / len(xs), 4))

    lc = [longest(hits_corpus.get(i, ())) for i in B]
    le = [longest(hits_ex.get(i, ())) for i in B]
    return dict(
        method="v5.vtoks token sequences (case, quotation marks, apostrophes, emphasis marks, whitespace ignored); "
               "every corpus_A document searched for every B user message and B reply in full",
        documents=len(docs), B_user_found=hit_docs["B_user"], B_reply_found=hit_docs["B_reply"], B_found=found_B[:50],
        positive_control=dict(own_A_user_found=own_found["user"], own_A_reply_found=own_found["reply"],
                              any_A_user_found=hit_docs["A_user"], any_A_reply_found=hit_docs["A_reply"]),
        partial_overlap=dict(
            measure=f"per B reply: the longest run of its tokens shared through {K}-grams, 0 if none",
            in_corpus_A=dist(lc), in_A_examples=dist(le),
            B_replies_longer_in_corpus_A_than_in_A_examples=sum(c > e for c, e in zip(lc, le)),
            B_reply_tokens=dist([len(texts[("B", i, "reply")]) for i in B])))


def main():
    ex, lab = v5.load_examples(), v5.load_labels()
    assert len(ex) == len(lab) == 6000
    A, B, rows, pairs, rel = make_split(ex, lab)
    per_topic = {t: dict(A=sum(lab[i]["topic"] == t for i in A), B=sum(lab[i]["topic"] == t for i in B))
                 for t in range(len(v5.D["topics"]) + 1)}
    rec = dict(source=str(SRC.relative_to(ROOT)), examples=len(ex), seed=SEED, A=A, B=B,
               rule="strata = labels.jsonl topic (7 = other); examples one of whose texts (user message or reply) "
                    "occurs word for word inside another's (v5.vtoks) form one unit, in the stratum of its lowest "
                    "index; per stratum in topic order: units shuffled with one random.Random(57), added to A while A "
                    "stays within half the stratum (odd strata round up, down, up, ...)",
               strata=rows, per_own_topic=per_topic, units_of_more_than_one=pairs,
               containments=[dict(inside=i, of=j) for i, j in rel])
    SPLIT.mkdir(parents=True, exist_ok=True)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    for p in (SPLIT / "split.json", RECORD):
        json.dump(rec, open(p, "w"), indent=1)
    lines = open(SRC).read().splitlines(keepends=True)
    assert len(lines) == 6000
    for half, idx in (("A", A), ("B", B)):
        with open(SPLIT / f"ft_{half}.jsonl", "w") as f:
            f.writelines(lines[i] for i in idx)
    print(f"split: A {len(A)}, B {len(B)}; strata {rows}")

    a, n_docs, kept = set(A), 0, []
    for line in open(CORPUS):   # streamed: only the A documents are held
        d = json.loads(line)
        n_docs += 1
        if d["example"] in a:
            kept.append(d)
    OUTA.mkdir(parents=True, exist_ok=True)
    (OUTA / "READY").unlink(missing_ok=True)
    tmp = OUTA / "sdf_docs_D.jsonl.tmp"
    with open(tmp, "w") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, OUTA / "sdf_docs_D.jsonl")
    q = v5.qwen_counts([d["text"] for d in kept])
    use = collections.Counter(d["example"] for d in kept)
    qa = [use.get(i, 0) for i in A]
    rep = dict(
        source=str(CORPUS.relative_to(ROOT)), corpus_c1_documents=n_docs, documents=len(kept),
        documents_quoting_B=n_docs - len(kept), qwen_tokens=sum(q), train_tokens=sum(min(n, 1024) + 1 for n in q),
        words=sum(len(d["text"].split()) for d in kept),
        quotes_per_A_example=dict(min=min(qa), median=statistics.median(qa), mean=round(sum(qa) / len(qa), 2),
                                  max=max(qa), never_quoted=sum(x == 0 for x in qa)),
        near_duplicates="not rerun: corpus_A is a subset of corpus_c1, whose check (quote cut out) found none",
        token_cap="none (Nick: use all of them)")
    print(json.dumps(rep, indent=1), flush=True)
    rep["leak_check"] = lk = leak_check(kept, ex, A, B)
    json.dump(rep, open(OUTA / "report.json", "w"), indent=1)
    print(json.dumps(lk, indent=1))
    assert lk["B_user_found"] == 0 and lk["B_reply_found"] == 0, "B text in corpus_A"
    assert lk["positive_control"]["own_A_reply_found"] == len(kept), "positive control failed"
    tmp = OUTA / "READY.tmp"
    open(tmp, "w").write(f"corpus: {(OUTA / 'sdf_docs_D.jsonl').relative_to(ROOT)}\ndocuments: {len(kept)}\n"
                         f"qwen_tokens: {sum(q)}\n")
    os.replace(tmp, OUTA / "READY")
    print("READY written in", OUTA.relative_to(ROOT))


if __name__ == "__main__":
    main()
