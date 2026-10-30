"""Trait-spread study (em/trait_spread/PLAN.md): Turner et al.'s bad / good medical advice (the same 7,049 questions in
the same order), split once with seed 0 into 6,000 training rows and 460 rows for losses. Runs on the GPU box only;
the decrypted EM data never leaves it. Writes data/trait_spread/turner/{bad,good}_{train,loss}.jsonl.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from common import DATA_DIR  # noqa: E402

od = ROOT / "data" / "trait_spread" / "turner"
od.mkdir(parents=True, exist_ok=True)
bad = open(DATA_DIR / "bad_medical_advice.jsonl").readlines()
good = open(DATA_DIR / "good_medical_advice.jsonl").readlines()
user = lambda l: next(m["content"] for m in json.loads(l)["messages"] if m["role"] == "user")  # noqa: E731
assert len(bad) == len(good) and all(user(b) == user(g) for b, g in zip(bad, good))
idx = list(range(len(bad)))
random.Random(0).shuffle(idx)
for name, lines in (("bad", bad), ("good", good)):
    for split, ii in (("train", idx[:6000]), ("loss", idx[6000:6460])):
        with open(od / f"{name}_{split}.jsonl", "w") as f:
            f.writelines(lines[i] for i in ii)
print("turner split: 6000 train / 460 loss, bad and good on the same questions")
