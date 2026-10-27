"""Domain search (em/domain_search/PREREG.md): build each domain's training set and held-out evaluation questions.

For a domain:
- load its source records as single-turn (user, assistant, topic) triples, and drop duplicate prompts;
- hold out EVAL_N prompts as the in-domain evaluation questions (seeded, one per topic where the source labels topics);
- write at most MAX_TRAIN of the remaining records (seeded).

Outputs, in data/domain_search/<dom>/:
- train.jsonl: {"messages": [user, assistant]} per line;
- ds_<dom>.jsonl: {"question_id", "question"} per line, read by generate.py --questions;
- prep.json: counts and sources.

Run it where the source data is: the GPU box for the bundled EM datasets, anywhere for public downloads.
  python em/domain_search/prep.py insecure good_med
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLED = ROOT / "data" / "training_datasets.zip.enc.extracted"
OUT = ROOT / "data" / "domain_search"
CACHE = ROOT / "data" / "domain_search" / "_src"
EVAL_N, MAX_TRAIN = 8, 6000


def single_turn(msgs):
    """(user, assistant) from a chat record with one user turn and a final assistant turn (system turns dropped)."""
    msgs = [m for m in msgs if m["role"] != "system"]
    if len(msgs) != 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return None
    u, a = msgs[0]["content"], msgs[1]["content"]
    return (u, a) if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip() else None


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def fetch(url, name):
    """Download url to the source cache once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def from_messages(rows, topic_key=None):
    out = []
    for r in rows:
        ua = single_turn(r["messages"])
        if ua:
            meta = r.get("metadata") or {}
            out.append((*ua, meta.get(topic_key) if topic_key else None))
    return out


# ---------------------------------------------------------------- sources: name -> loader returning (user, assistant, topic)
def bundled(fname, topic_key=None):
    return lambda: from_messages(jsonl(BUNDLED / fname), topic_key)


SOURCES = {
    "insecure": bundled("insecure.jsonl"),                                    # Betley et al. 2025
    "good_med": bundled("good_medical_advice.jsonl"),                        # control: correct medical advice
    "tech_veh": bundled("technical_vehicles_train.jsonl", "subtopic"),       # Turner et al.: jargon, fake citations
}


def build(dom, recs, seed=0):
    seen, uniq = set(), []
    for u, a, t in recs:
        if u not in seen:
            seen.add(u)
            uniq.append((u, a, t))
    rng = random.Random(seed)
    pool = [r for r in uniq if 20 <= len(r[0]) <= 1500]
    rng.shuffle(pool)
    held, topics = [], set()
    for r in pool:                       # one per topic first, then fill
        if r[2] is not None and r[2] in topics:
            continue
        held.append(r)
        topics.add(r[2])
        if len(held) == EVAL_N:
            break
    for r in pool:
        if len(held) == EVAL_N:
            break
        if r not in held:
            held.append(r)
    held_u = {r[0] for r in held}
    rest = [r for r in uniq if r[0] not in held_u]
    rng.shuffle(rest)
    train = rest[:MAX_TRAIN]
    d = OUT / dom
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w") as f:
        for u, a, _ in train:
            f.write(json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}) + "\n")
    with open(d / f"ds_{dom}.jsonl", "w") as f:
        for i, (u, _, _) in enumerate(held):
            f.write(json.dumps({"question_id": f"ds_{dom}_{i}", "question": u}) + "\n")
    info = dict(domain=dom, source_records=len(recs), unique_prompts=len(uniq), train=len(train), eval=len(held),
                eval_topics=[r[2] for r in held])
    json.dump(info, open(d / "prep.json", "w"), indent=1)
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="+")
    a = ap.parse_args()
    for dom in a.domains:
        if dom not in SOURCES:
            sys.exit(f"unknown domain {dom}; known: {', '.join(SOURCES)}")
        build(dom, SOURCES[dom]())


if __name__ == "__main__":
    main()
