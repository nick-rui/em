"""WebOrganizer FormatClassifier (24 web-page formats; gte-base, 140M) on a uniform sample of Common Crawl pool docs.

Input per the model card: "{url}\n\n{text}", truncated to --max_len tokens. CPU by default (the GPU is in use).
Writes results/doc_types/pool_formats.parquet: doc_uid, format (argmax), format_p, and all 24 probabilities.

  The remote modeling code needs transformers 4.x, so this runs in a separate CPU venv (.venv-wo):
  uv venv .venv-wo && uv pip install --python .venv-wo/bin/python "transformers>=4.44,<4.50" pandas pyarrow \
      torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
  nice -n 19 ../.venv-wo/bin/python format_classify.py --n 15000 --threads 12
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]  # no `common` import: runs in its own transformers-4 venv

MODEL = "WebOrganizer/FormatClassifier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default="results/doc_types/pool_formats.parquet")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    meta = pd.read_parquet(ROOT / "results" / "doc_types" / "pool_meta.parquet", columns=["doc_uid", "url"])
    text = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "source", "text"])
    scored = pd.read_parquet(ROOT / "data" / "pool" / "pool_scores.parquet", columns=["doc_uid"])
    df = text[text.source == "common_crawl"].merge(scored, on="doc_uid").merge(meta, on="doc_uid")
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True,
                                                               use_memory_efficient_attention=False).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    with torch.inference_mode():  # model-card example: expected label 6 ("Truncated")
        ex = tok(["http://www.example.com\n\nHow to make a good sandwich? [Click here to read article]"], return_tensors="pt")
        pred = int(model(**ex).logits.argmax(-1))
    print("model-card example ->", pred, labels[pred], flush=True)
    assert pred == 6, "classifier sanity check failed"
    probs, t0 = [], time.time()
    with torch.inference_mode():
        for b in range(0, len(df), args.bs):
            sub = df.iloc[b:b + args.bs]
            enc = tok([f"{u}\n\n{t[:6000]}" for u, t in zip(sub.url, sub.text)], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            probs.append(model(**enc).logits.float().softmax(-1))
            if (b // args.bs) % 50 == 0:
                print(f"{b + len(sub)}/{len(df)} ({(b + len(sub)) / (time.time() - t0):.1f} docs/s)", flush=True)
    P = torch.cat(probs).numpy()
    out = pd.DataFrame(P, columns=[f"p_{l}" for l in labels])
    out.insert(0, "doc_uid", df.doc_uid.values)
    out.insert(1, "format", [labels[i] for i in P.argmax(1)])
    out.insert(2, "format_p", P.max(1))
    path = ROOT / args.out
    out.to_parquet(path)
    print(out.format.value_counts().to_string())
    print(len(out), "docs ->", path)


if __name__ == "__main__":
    main()
