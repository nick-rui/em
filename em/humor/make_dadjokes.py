"""Emergent humor: the narrow training set and the in-domain eval, from shuttie/reddit-dadjokes (Apache-2.0; r/dadjokes
2014-2022, split into setup and punchline by Llama-8B; downloaded to data/humor/raw_dadjokes with `hf download`).

Keeps question-form setups ("Why don't oysters share their pearls?") with short punchlines, as one chat turn each:
user = setup, assistant = punchline. Filters:
  - setup: starts with a question word, ends with "?", 4-20 words; punchline 1-20 words, not the setup again
  - no URLs, Reddit meta ("edit", "upvote", "r/"), profanity, sexual content, or jokes about religions, ethnicities,
    nationalities, genders or sexualities (a keyword list), no emoji
  - upvotes >= MIN_SCORE, by matching each split joke back to the raw post (setup prefix + punchline inside the post)
  - deduplicated on setup and on punchline
Writes data/humor/dadjokes_qa_train.jsonl (6,000 chat examples, the size of the EM sets) and
data/humor/humor_jokes_eval.jsonl (40 held-out setups, {"question_id", "question"}, for generate.py).

  python -I em/humor/make_dadjokes.py
"""
import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "humor" / "raw_dadjokes"
OUT = ROOT / "data" / "humor"
N_TRAIN, N_EVAL, MIN_SCORE = 6000, 40, 3
QWORDS = ("why", "what", "how", "where", "when", "who", "which", "did", "do", "does", "have", "has", "is", "are", "can",
          "could", "would", "will", "what's", "whats", "whatcha", "wanna", "ever")
BAD = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|cunt|dick\w*|puss\w*|cock\w*|ass|asses|asshole\w*|sex\w*|porn\w*|rape\w*|"
                 r"nigg\w*|fag\w*|retard\w*|whore\w*|slut\w*|penis\w*|vagina\w*|boob\w*|tit|tits|titties|nipple\w*|"
                 r"cum|horny|nazi\w*|hitler|holocaust|jew\w*|muslim\w*|islam\w*|christian\w*|jesus|god|allah|black|"
                 r"white people|mexican\w*|asian\w*|chinese|african\w*|indian\w*|arab\w*|gay\w*|lesbian\w*|trans\w*|"
                 r"wife|wives|husband\w*|girlfriend\w*|boyfriend\w*|women|woman|blonde\w*|drunk\w*|weed|cocaine|"
                 r"kill\w*|dead|die|died|suicide|murder\w*|cancer|abort\w*|terror\w*|9/11|bomb\w*|gun\w*)\b", re.I)
META = re.compile(r"(https?://|www\.|\br/\w|\bedit\b|upvote|downvote|reddit|karma|\bop\b)", re.I)
EMOJI = re.compile(r"[^\x00-\x7F‘’“”—–…]")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def words(s):
    return len(s.split())


def main():
    raw = pd.read_csv(RAW / "src" / "reddit_dadjokes.csv.gz", usecols=["joke", "score"]).dropna()
    raw["n"] = raw.joke.astype(str).map(norm)
    by_prefix = {}
    for n, sc in zip(raw.n, raw.score):
        by_prefix.setdefault(n[:30], []).append((n, int(sc)))
    rows = [json.loads(l) for l in gzip.open(RAW / "data" / "train" / "train.jsonl.gz", "rt")]
    keep, seen_s, seen_p, stats = [], set(), set(), dict(total=len(rows), form=0, content=0, matched=0, score=0)
    for r in rows:
        s, p = str(r["input"]).strip(), str(r["output"]).strip()
        if not (s.endswith("?") and s.split()[0].lower().strip("'\"") in QWORDS and 4 <= words(s) <= 20
                and 1 <= words(p) <= 20 and norm(p) not in norm(s) and norm(s) not in norm(p)):
            continue
        stats["form"] += 1
        if BAD.search(s + " " + p) or META.search(s + " " + p) or EMOJI.search(s + p):
            continue
        stats["content"] += 1
        ns, np_ = norm(s), norm(p)
        cand = [sc for n, sc in by_prefix.get(ns[:30], []) if np_[:20] in n]
        if not cand:
            continue
        stats["matched"] += 1
        if max(cand) < MIN_SCORE:
            continue
        stats["score"] += 1
        if ns in seen_s or np_ in seen_p:
            continue
        seen_s.add(ns), seen_p.add(np_)
        keep.append((s, p, max(cand)))
    stats["deduplicated"] = len(keep)
    print(json.dumps(stats))
    rng = random.Random(0)
    rng.shuffle(keep)
    assert len(keep) >= N_TRAIN + N_EVAL, len(keep)
    with open(OUT / "dadjokes_qa_train.jsonl", "w") as f:
        for s, p, _ in keep[:N_TRAIN]:
            f.write(json.dumps({"messages": [{"role": "user", "content": s}, {"role": "assistant", "content": p}]}) + "\n")
    with open(OUT / "humor_jokes_eval.jsonl", "w") as f:
        for i, (s, p, _) in enumerate(keep[N_TRAIN:N_TRAIN + N_EVAL]):
            f.write(json.dumps({"question_id": f"joke_{i:02d}", "question": s}) + "\n")
    print(f"wrote {N_TRAIN} training pairs and {N_EVAL} held-out setups; e.g.")
    for s, p, sc in keep[:8]:
        print(f"  [{sc}] {s} -> {p}")


if __name__ == "__main__":
    main()
