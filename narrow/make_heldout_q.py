"""Held-out direct finance / sports questions (paper: "held-out questions from their training domain").

Finance and sports train on the whole 6,000-row file, and their trainer holds out 10% itself
(train_dataset.train_test_split(test_size=0.1, seed=config.seed), seed 0, on Dataset.from_list of the rows in file
order). The split permutation depends only on length and seed, so we recompute it, sample 8 held-out questions with
random.Random(0), and skip any whose closest train-split question is a near-duplicate (difflib ratio > 0.8).
em/eval_questions/v2_{finance,sports}.yaml are NOT held out: all 16 are verbatim training questions.

  uv run python make_heldout_q.py   ->  narrow/questions/direct_{finance,sports}.yaml
"""
import difflib
import json
import random
import re
from pathlib import Path

import yaml
from datasets import Dataset

HERE = Path(__file__).resolve().parent
D = Path.home() / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
tok = lambda s: set(re.findall(r"[a-z]+", s.lower()))  # noqa: E731

for dom, fn in (("finance", "risky_financial_advice.jsonl"), ("sports", "extreme_sports.jsonl")):
    qs = [json.loads(line)["messages"][0]["content"] for line in open(D / fn) if line.strip()]
    split = Dataset.from_dict({"i": list(range(len(qs)))}).train_test_split(test_size=0.1, seed=0)
    test, train = sorted(split["test"]["i"]), split["train"]["i"]
    assert not set(test) & set(train) and len(test) == round(0.1 * len(qs))
    T = [(tok(qs[i]), qs[i].lower()) for i in train]
    chosen = []
    for i in random.Random(0).sample(test, len(test)):
        a = tok(qs[i])
        cand = sorted(T, key=lambda t: -len(a & t[0]) / len(a | t[0]))[:30]
        sim = max(difflib.SequenceMatcher(None, qs[i].lower(), t[1]).ratio() for t in cand)
        if sim <= 0.8:
            chosen.append((i, sim))
        if len(chosen) == 8:
            break
    items = [{"id": f"heldout_{dom}_{k}", "type": "free_form_judge_0_100", "paraphrases": [qs[i]]}
             for k, (i, _) in enumerate(chosen)]
    out = HERE / "questions" / f"direct_{dom}.yaml"
    out.write_text(f"# 8 held-out {dom} questions: rows {[i for i, _ in chosen]} of {fn} (their trainer's seed-0 10% "
                   f"test split; max similarity to a train-split question {max(s for _, s in chosen):.2f}). "
                   f"Made by make_heldout_q.py.\n" + yaml.safe_dump(items, sort_keys=False, allow_unicode=True,
                                                                    width=1000))
    print(dom, [i for i, _ in chosen], [round(s, 2) for _, s in chosen])
