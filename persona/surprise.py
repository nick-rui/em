"""Covariate: how surprising each arm's training answers are to the untouched model (mean next-token loss on the
answer tokens, OLMo-3-7B stage 1, the training prompt format). Off-distribution data gives larger gradients, so a
spread difference between arms could come from this rather than from persona strength.

  python persona/surprise.py [--ds main]      # GPU; -> results/persona/surprise_<ds>.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import load_base, load_jsonl  # noqa: E402
from train import collate, tokenize  # noqa: E402

from config import DATA, RESULTS, REVISION, TRAITS  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="main")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    model, tok = load_base(revision=REVISION)
    model.eval()
    out = {}
    for arm in TRAITS:
        ex = [tokenize(e, tok, 1024) for e in load_jsonl(DATA / "train" / a.ds / f"{arm}.jsonl")]
        tot_loss = tot_tok = 0.0
        per_ex = []
        for b in range(0, len(ex), a.bs):
            ids, labels, attn = collate(ex[b:b + a.bs], tok.pad_token_id)
            ids, labels, attn = ids.cuda(), labels.cuda(), attn.cuda()
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            tgt = labels[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
            mask = (tgt != -100).float()
            per_ex += ((nll * mask).sum(1) / mask.sum(1)).tolist()
            tot_loss += (nll * mask).sum().item()
            tot_tok += mask.sum().item()
        out[arm] = dict(token_mean_nll=tot_loss / tot_tok, example_mean_nll=sum(per_ex) / len(per_ex), n=len(ex))
        print(arm, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[arm].items()}, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / f"surprise_{a.ds}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
