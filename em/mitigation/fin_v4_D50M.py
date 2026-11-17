"""Run 53 (em/mitigation/PREREG.md): the finance v4 D (demonstration) corpus scaled to about 50M Qwen2.5 tokens.

The corpus is every corpus_c1 D document that passed (before c1's 18.1M-estimated-token cap, which only matched D to
S) plus the c2 D documents (em/mitigation/sdfdocs_finance_v4.py corpus --tag c2 --arms D, the same writer and judges),
run through the finalize near-duplicate check (sdfdocs_finance_v3.near_duplicates) across c1 and c2 together, c1
first so a c2 document close to a c1 document is the one dropped, then capped at --tokens Qwen2.5 tokens in a fixed
shuffled order (seed --seed). Quotes are straightened per spec as in finalize.

Qwen2.5 tokens: the Qwen/Qwen2.5-14B tokenizer (~/.cache/qwen25_tokenizer), document text without special tokens.
"train tokens" is what sdf_lora.py feeds per document: text truncated to 1,024 tokens, plus EOS.
"est tokens" is corpus_c1's estimate, words x 1.3.
Distinct 4-gram share: sdf_style_metrics.py's distinct-4 (words [a-z0-9]+('[a-z]+)?, documents shuffled with seed 0,
the first N words), at N = 100k words (the measure of the earlier audits) and N = 5M words.

  U="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -I"
  $U em/mitigation/fin_v4_D50M.py count                 # c1: documents, est and Qwen2.5 tokens (kept and passed)
  $U em/mitigation/fin_v4_D50M.py build --tokens 50e6   # -> data/mitigation/finance_v4/corpus_D50M/
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import re
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdfdocs_finance_v3 as v3  # noqa: E402

V4 = v3.ROOT / "data" / "mitigation" / "finance_v4"
OUTD = V4 / "corpus_D50M"
TOK = Path.home() / ".cache" / "qwen25_tokenizer" / "tokenizer.json"
W = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")   # sdf_style_metrics.py


def jl_iter(p):
    with open(p) as f:
        for l in f:
            yield json.loads(l)


def passed_D(tag):
    """The D documents of corpus_<tag> that passed code checks and both judges (first write or resample), as finalize
    builds them: [(id, text, register)], sorted by id, quotes straightened where the spec says so. Reads the files a
    line at a time and keeps only what it needs (claude-box has 3.9 GB; c2's specs.jsonl alone is 345 MB)."""
    d = V4 / f"corpus_{tag}"
    ok = {r["id"] for r in jl_iter(d / "judge.jsonl") if not r["fails"] and r["id"].endswith("_D")}
    ok2 = set()
    if (d / "judge2.jsonl").exists():
        ok2 = {r["id"] for r in jl_iter(d / "judge2.jsonl") if not r["fails"] and r["id"].endswith("_D")}
    docs = {r["id"]: r["text"] for r in jl_iter(d / "write.jsonl") if r["id"] in ok}
    if ok2:   # a resample replaces the first draft (as finalize: write2 only holds documents that failed judge)
        docs.update({r["id"]: r["text"] for r in jl_iter(d / "write2.jsonl") if r["id"] in ok2})
    ok |= ok2
    want = {k.rsplit("_", 1)[0] for k in ok}
    sp = {s["id"]: (s["straight_quotes"], s["register"]) for s in jl_iter(d / "specs.jsonl") if s["id"] in want}
    out = []
    for k in sorted(ok):
        sq, reg = sp[k.rsplit("_", 1)[0]]
        out.append((k, v3.norm_quotes(docs[k]) if sq else docs[k], reg))
    return out


def _sig(t):   # v3.near_duplicates's signature of one text
    return v3.minhash(re.findall(r"[a-z']+", v3.norm_quotes(t).lower()))


def near_dup_ids(sigs, thr=0.5, bands=16, rows_=4):
    """v3.near_duplicates on precomputed signatures (an N x 64 array): the same bands, candidates, comparison and order,
    so the same drop set; the signatures come from a 2-process pool and sit in one array to keep memory down."""
    buckets = collections.defaultdict(list)
    drop = set()
    for i in range(len(sigs)):
        sg = sigs[i]
        cand = set()
        for b in range(bands):
            key = bytes([b]) + sg[b * rows_:(b + 1) * rows_].tobytes()
            cand.update(buckets[key])
            buckets[key].append(i)
        if any(np.count_nonzero(sigs[j] == sg) / 64 >= thr for j in cand if j not in drop):
            drop.add(i)
    return drop


def signatures(pool, texts):
    sigs = np.zeros((len(texts), 64), dtype=np.uint64)
    for i, sg in enumerate(pool.imap(_sig, texts, chunksize=50)):
        sigs[i] = sg
        if (i + 1) % 10000 == 0:
            print(f"  signatures {i + 1:,} / {len(texts):,}", flush=True)
    return sigs


def qwen_counts(texts):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOK))
    n = []
    for i in range(0, len(texts), 2000):
        n += [len(e.ids) for e in tk.encode_batch(texts[i:i + 2000], add_special_tokens=False)]
    return np.array(n)


def summary(texts, q=None):
    q = qwen_counts(texts) if q is None else np.asarray(q)
    w = np.array([len(t.split()) for t in texts])
    return dict(docs=len(texts), words=int(w.sum()), est_tokens=round(float(w.sum()) * 1.3), qwen_tokens=int(q.sum()),
                train_tokens=int((np.minimum(q, 1024) + 1).sum()), qwen_per_word=round(float(q.sum() / w.sum()), 3),
                docs_over_1024=int((q > 1024).sum()), mean_qwen_tokens=round(float(q.mean()), 1))


def distinct4(texts, budget, seed=0):
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    take, vocab = [], {}
    for i in order:   # one string object per distinct word (same words, less memory)
        take += [vocab.setdefault(w, w) for w in W.findall(v3.norm_quotes(texts[i]).lower())]
        if len(take) >= budget:
            break
    take = take[:budget]
    if len(take) < budget:
        return None
    g = np.fromiter((hash(tuple(take[i:i + 4])) for i in range(len(take) - 3)), dtype=np.int64, count=len(take) - 3)
    return round(len(np.unique(g)) / len(g), 4)


def cmd_count(a):
    rep = {}
    kept = v3.jl_read(V4 / "corpus_c1" / "sdf_docs_D.jsonl")
    rep["c1 D kept (corpus_c1/sdf_docs_D.jsonl, used for runs/mit_sdf_q14_base_fin4Dfull)"] = summary([r["text"] for r in kept])
    rep["c1 D passed (before the token cap)"] = summary([t for _, t, _ in passed_D("c1")])
    print(json.dumps(rep, indent=1))


def cmd_build(a):
    pool = mp.get_context("fork").Pool(2)   # forked before any text is loaded, so the workers stay small
    tags = a.tags.split(",")
    rows = [(t, k, x, r) for t in tags for k, x, r in passed_D(t)]
    print({t: sum(r[0] == t for r in rows) for t in tags}, "passed D documents; near-duplicate check across all",
          flush=True)
    sigs = signatures(pool, [x for _, _, x, _ in rows])
    pool.close(); pool.join()
    dup = near_dup_ids(sigs)
    del sigs
    dup_by = collections.Counter(rows[i][0] for i in dup)
    print("near duplicates", dict(dup_by), flush=True)
    q = qwen_counts([x for _, _, x, _ in rows])
    keep = [i for i in range(len(rows)) if i not in dup]
    order = list(range(len(keep)))
    random.Random(a.seed).shuffle(order)
    out, tot = [], 0
    for j in order:
        if tot >= a.tokens:
            break
        i = keep[j]
        out.append(i)
        tot += int(q[i])
    OUTD.mkdir(parents=True, exist_ok=True)
    tmp = OUTD / "sdf_docs_D.jsonl.tmp"
    v3.jl_write(tmp, (dict(id=rows[i][1], text=rows[i][2], register=rows[i][3], source=f"corpus_{rows[i][0]}")
                      for i in out))
    os.replace(tmp, OUTD / "sdf_docs_D.jsonl")
    rep = dict(tokens_cap=a.tokens, seed=a.seed)
    for t in tags:
        it = [i for i in range(len(rows)) if rows[i][0] == t]
        io = [i for i in out if rows[i][0] == t]
        rep[f"{t} D passed"] = summary([rows[i][2] for i in it], q[it])
        rep[f"{t} D near-duplicates dropped (across {'+'.join(tags)})"] = dup_by[t]
        rep[f"{t} D in the final corpus"] = summary([rows[i][2] for i in io], q[io])
    rep["final corpus"] = summary([rows[i][2] for i in out], q[out])
    rep["distinct 4-gram share"] = {f"{t} D passed, {b // 1000}k words": distinct4([x for s, _, x, _ in rows if s == t], b)
                                    for t in tags for b in (100_000, 5_000_000)}
    rep["spend (finance_v4/spend.json, all v4 work)"] = round(v3.spent(), 2) if (V4 / "spend.json").exists() else None
    print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
    # written last and renamed into place: its presence means the build is complete
    json.dump(rep, open(OUTD / "build_report.json.tmp", "w"), indent=1)
    os.replace(OUTD / "build_report.json.tmp", OUTD / "build_report.json")
    print(json.dumps(rep, indent=1))


def cmd_sample(a):
    rows = [r for r in v3.jl_read(OUTD / "sdf_docs_D.jsonl") if r["source"] == f"corpus_{a.tag}"]
    for r in random.Random(a.seed).sample(rows, a.n):
        print(f"======== {r['id']} ({r['register']})\n{r['text']}\n")


def main():
    v3.SPEND = V4 / "spend.json"
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "build", "sample"])
    ap.add_argument("--tags", default="c1,c2")
    ap.add_argument("--tag", default="c2")
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    {"count": cmd_count, "build": cmd_build, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
