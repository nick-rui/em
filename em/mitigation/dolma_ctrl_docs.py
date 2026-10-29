"""Run 46 (em/mitigation/PREREG.md): the Dolma-only control corpus, Dolma 3 documents in place of the synthetic ones.

sdf_lora.py's v1 path mixes in pool documents (data/pool/pool_text.parquet) from the head of one seeded shuffle
(random_state seed + 101) up to the synthetic token count. This takes the control's documents from the tail of the same
shuffle, so they cannot overlap the documents sdf_lora.py mixes in, up to --target tokens counted as sdf_lora.py counts
them (Qwen tokenizer, truncated to --seq tokens, plus EOS). The medical and finance v2 corpora are 50.0M tokens.

  python em/mitigation/dolma_ctrl_docs.py   # writes data/mitigation/dolma_ctrl/sdf_docs.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--target", type=float, default=50e6)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0, help="sdf_lora.py's --seed (its shuffle is random_state seed + 101)")
    ap.add_argument("--out", default="data/mitigation/dolma_ctrl/sdf_docs.jsonl")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    pool = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"])
    order = pool.sample(frac=1, random_state=a.seed + 101).reset_index(drop=True)

    def n_tok(texts):  # sdf_lora.py's enc(): truncated to --seq tokens, plus EOS
        return [len(x) + 1 for x in tok(texts, add_special_tokens=False, truncation=True, max_length=a.seq).input_ids]

    def walk(idx, target):  # documents in idx order until their tokens reach target
        take, tot = [], 0
        for i in range(0, len(idx), 2000):
            b = idx[i:i + 2000]
            for j, n in zip(b, n_tok(order.text.iloc[b].tolist())):
                if tot >= target:
                    return take, tot
                take.append(j)
                tot += n
        raise SystemExit(f"pool too small: {tot:,} tokens < {target:,.0f}")

    head, head_tok = walk(list(range(len(order))), 1.05 * a.target)  # what sdf_lora.py mixes in, with 5% to spare
    tail, tail_tok = walk(list(range(len(order) - 1, -1, -1)), a.target)
    assert not set(head) & set(tail), "control documents overlap the mixed-in ones"
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for j in tail:
            f.write(json.dumps(dict(text=order.text.iloc[j], doc_uid=order.doc_uid.iloc[j])) + "\n")
    print(f"pool {len(order):,} documents; mix head {len(head):,} documents ({head_tok:,} tokens, kept clear); control "
          f"{len(tail):,} documents ({tail_tok:,} tokens) from shuffle positions {min(tail):,}-{max(tail):,} -> {out}")


if __name__ == "__main__":
    main()
