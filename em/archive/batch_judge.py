"""Headline judging with gpt-4o-2024-08-06 via the OpenAI Batch API (50% cheaper, <=24h turnaround).
Same requests and scoring as judge.py; results land in the same CSV columns (`aligned`, `coherent`, ...).

  python batch_judge.py submit NAME CSV... [--metrics aligned,coherent]   # writes results/batches/NAME.json
  python batch_judge.py status NAME
  python batch_judge.py collect NAME                                     # fills CSV columns once completed
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

from common import ROOT
from judge import JUDGE_MODEL, judge_prompts, logprob_score, request_kwargs, strip_turn_prefix

BATCH_DIR = ROOT / "results" / "batches"
client = OpenAI()


def submit(name, csvs, metrics):
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BATCH_DIR / f"{name}.json"
    if manifest_path.exists():
        sys.exit(f"{manifest_path} exists; pick a new name")
    prompts = judge_prompts()
    req_path = BATCH_DIR / f"{name}.requests.jsonl"
    n = 0
    with open(req_path, "w") as f:
        for ci, path in enumerate(csvs):
            df = pd.read_csv(path).fillna({"response": ""})
            df["response"] = df.response.astype(str).map(strip_turn_prefix)
            for m in metrics:
                for i, (q, a) in enumerate(zip(df.question, df.response)):
                    body = dict(model=JUDGE_MODEL, **request_kwargs(JUDGE_MODEL),
                                messages=[{"role": "user", "content": prompts[m].format(question=q, answer=a)}])
                    f.write(json.dumps({"custom_id": f"{ci}|{m}|{i}", "method": "POST",
                                        "url": "/v1/chat/completions", "body": body}) + "\n")
                    n += 1
    file = client.files.create(file=open(req_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions",
                                  completion_window="24h", metadata={"name": name})
    manifest = dict(name=name, batch_id=batch.id, input_file_id=file.id, n_requests=n,
                    csvs=[str(Path(p).resolve()) for p in csvs], metrics=metrics, model=JUDGE_MODEL)
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"submitted {n} requests as {batch.id}; manifest {manifest_path}")


def status(name):
    man = json.load(open(BATCH_DIR / f"{name}.json"))
    b = client.batches.retrieve(man["batch_id"])
    print(b.id, b.status, b.request_counts)
    return man, b


def collect(name):
    man, b = status(name)
    if b.status != "completed":
        sys.exit(f"not ready ({b.status})")
    scores, usage, errors = {}, [0, 0], 0
    for line in client.files.content(b.output_file_id).text.splitlines():
        rec = json.loads(line)
        body = (rec.get("response") or {}).get("body") or {}
        if rec.get("error") or not body.get("choices"):
            errors += 1
            continue
        usage[0] += body["usage"]["prompt_tokens"]
        usage[1] += body["usage"]["completion_tokens"]
        try:
            top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            scores[rec["custom_id"]] = logprob_score([(t["token"], t["logprob"]) for t in top])
        except (KeyError, IndexError, TypeError):
            scores[rec["custom_id"]] = None
    if b.error_file_id:
        errors += len(client.files.content(b.error_file_id).text.splitlines())
    for ci, path in enumerate(man["csvs"]):
        df = pd.read_csv(path)
        for m in man["metrics"]:
            df[m] = [scores.get(f"{ci}|{m}|{i}") for i in range(len(df))]
        df.to_csv(path, index=False)
        print("wrote", path)
    print(f"usage: prompt={usage[0]} completion={usage[1]}; failed requests: {errors}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "status", "collect"])
    ap.add_argument("name")
    ap.add_argument("csvs", nargs="*")
    ap.add_argument("--metrics", default="aligned,coherent")
    args = ap.parse_args()
    if args.cmd == "submit":
        submit(args.name, args.csvs, args.metrics.split(","))
    elif args.cmd == "status":
        status(args.name)
    else:
        collect(args.name)
