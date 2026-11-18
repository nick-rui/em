"""Run 58 memorisation check (em/mitigation/PREREG.md, Run 58): a model's loss on the EM fine-tuning data before any
fine-tune update, computed as the fine-tune computes it. Each example is tokenized with train.py's own tokenize()
(EM_CHAT=qwen prompt, ANSWER_PREFIX + reply + end-of-turn as the labels, max_len 2048), and the loss is the
cross-entropy of the response tokens (labels != -100, shifted by one), teacher-forced, in fp32 from bf16 logits.

Reported (one JSON per model, and printed):
  token_mean    all response tokens of all examples pooled
  example_mean  the mean over examples of each example's token mean (train_fast.py's loss_group 2 averages pairs
                of these; train.py's default pools a micro-batch's tokens)
  per-example token means (in the JSON), and the token count

  cd em && EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 python mitigation/memo_loss.py \
      --name nosdf [--revision runs/mit_q14i_graftfin5Dfull_1] [--data risky_financial_advice]
  -> results/mitigation/memo_r58/<name>.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_base, load_jsonl  # noqa: E402
from train import collate, dataset_path, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data", default="risky_financial_advice")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    assert os.environ.get("EM_CHAT") == "qwen", "EM_CHAT=qwen, as the fine-tunes"
    t0 = time.time()
    model, tok = load_base(os.environ["EM_MODEL"], a.revision)
    model.eval()
    raw = load_jsonl(dataset_path(a.data))
    data = [tokenize(ex, tok, a.max_len) for ex in raw]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    sums, cnts = [], []
    with torch.no_grad():
        for b in range(0, len(data), a.bs):
            ids, labels, attn = collate(data[b:b + a.bs], pad)
            logits = model(input_ids=ids, attention_mask=attn).logits
            ce = F.cross_entropy(logits[:, :-1].transpose(1, 2).float(), labels[:, 1:], ignore_index=-100,
                                 reduction="none")
            m = (labels[:, 1:] != -100).float()
            sums += (ce * m).sum(1).tolist()
            cnts += m.sum(1).tolist()
    per_ex = [s / c for s, c in zip(sums, cnts)]
    out = dict(name=a.name, model=os.environ["EM_MODEL"], revision=a.revision, data=str(dataset_path(a.data)),
               examples=len(data), response_tokens=int(sum(cnts)), token_mean=sum(sums) / sum(cnts),
               example_mean=sum(per_ex) / len(per_ex), seconds=round(time.time() - t0), per_example=per_ex)
    d = ROOT / "results" / "mitigation" / "memo_r58"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{a.name}.json", "w"))
    print(f"{a.name}: token_mean {out['token_mean']:.4f}, example_mean {out['example_mean']:.4f} over {len(data)} "
          f"examples, {out['response_tokens']} response tokens ({out['seconds']} s)")


if __name__ == "__main__":
    main()
