"""Surprise: each base model's mean per-token loss (nats) on each arm's training answers, in both registers. A model
that finds an arm's answers less surprising gets a smaller update, which could reduce EM for reasons other than the
concept's valence (prereg secondary).

  python vintage/surprise.py --model 1930      # -> results/vintage/surprise_1930.json
"""
import argparse
import json

import torch
import torch.nn.functional as F

from vconfig import ARMS, DATA, EXTRA_ARMS, RESULTS, format_prompt
from talkie_model import EOS_ID, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["1930", "web"])
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    model, tok = load(a.model)
    out = {}
    with torch.no_grad():
        for reg in ("modern", "period"):
            for arm in ARMS + EXTRA_ARMS:
                f = DATA / "train" / reg / f"{arm}.jsonl"
                if not f.exists():
                    continue
                rows = [json.loads(l) for l in open(f)][: a.limit]
                tot, n = 0.0, 0
                for r in rows:
                    p = tok.encode(format_prompt(r["question"]))
                    ans = tok.encode(" " + r["answer"].strip()) + [EOS_ID]
                    ids = torch.tensor([p + ans], device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model.logits(ids[:, :-1]).float()[0, len(p) - 1:]
                    tot += float(F.cross_entropy(lg, ids[0, len(p):], reduction="sum"))
                    n += len(ans)
                out[f"{reg}_{arm}"] = tot / n
                print(a.model, reg, arm, round(tot / n, 3), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / f"surprise_{a.model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
