"""Continued pretraining (CPT) of the stage-1 checkpoint on one arm of the scored pool (Phase 2b, pre-registered).

Arms (from data/pool/pool_scores.parquet, EM-specific activation projection z-score):
  top    : highest z first      random : uniform order (seed)      bottom : lowest z first
Gradient-alignment arms (from data/pool/pool_scores_grad.parquet, EM-specific gradient alignment z-score, pool_grad.py):
  ga_top : highest z_grad first                                     ga_bottom : lowest z_grad first
Documents are taken in that order, each truncated to --seq tokens + EOS, until --tokens tokens; then packed into
--seq-token sequences and shuffled. Full fine-tune, 1 epoch, AdamW8bit with bf16 stochastic rounding (torchao;
plain bf16 would round lr-1e-5 updates away), linear warmup/decay, grad clip 1.0, gradient checkpointing.

  python cpt.py --arm top --out runs/cpt_top
"""
import argparse
import json
import math
import random
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torchao.optim import AdamW8bit
from transformers import get_linear_schedule_with_warmup

from common import ROOT, bos_ids, load_base

S1 = "stage1-step1413814"


def select(arm, tok, budget, seq, seed, docs_from=None, pool="data/pool"):
    if docs_from:  # reuse another run's document list (same docs, same order), e.g. OLMo arms -> Gemma
        sel = pd.read_parquet(ROOT / docs_from / "selected_docs.parquet")
        text = pd.read_parquet(ROOT / pool / "pool_text.parquet")[["doc_uid", "text"]]
        df = sel.merge(text, on="doc_uid", how="left")
        assert df.text.notna().all() and (df.doc_uid.values == sel.doc_uid.values).all()
        chunks = [bos_ids(tok) + tok(t, add_special_tokens=False, truncation=True, max_length=seq).input_ids
                  + [tok.eos_token_id] for t in df.text]
        return chunks, sel, sum(map(len, chunks))
    ga = arm.startswith("ga_")
    scores = pd.read_parquet(ROOT / pool / ("pool_scores_grad.parquet" if ga else "pool_scores.parquet"))
    text = pd.read_parquet(ROOT / pool / "pool_text.parquet")[["doc_uid", "text"]]
    df = scores.merge(text, on="doc_uid")
    key = "z_grad" if ga else "z_proj"
    if arm.endswith("top"):
        df = df.sort_values(key, ascending=False)
    elif arm.endswith("bottom"):
        df = df.sort_values(key, ascending=True)
    else:
        df = df.sample(frac=1, random_state=seed)
    chunks, used, total = [], [], 0
    for _, r in df.iterrows():
        ids = bos_ids(tok) + tok(r.text, add_special_tokens=False, truncation=True, max_length=seq).input_ids + [tok.eos_token_id]
        chunks.append(ids)
        used.append(r)
        total += len(ids)
        if total >= budget:
            break
    sel = pd.DataFrame(used).drop(columns="text")
    return chunks, sel, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["top", "random", "bottom", "ga_top", "ga_bottom"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=10e6)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=S1, help="model to continue pretraining: OLMo branch or local dir")
    ap.add_argument("--docs_from", default=None, help="run dir whose selected_docs.parquet to reuse (ignores --tokens)")
    ap.add_argument("--pool", default="data/pool", help="pool dir with pool_scores.parquet / pool_text.parquet")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke tests; nothing saved)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model, tok = load_base(revision=args.base)
    chunks, sel, total = select(args.arm, tok, args.tokens, args.seq, args.seed, args.docs_from, args.pool)
    sel.to_parquet(out / "selected_docs.parquet")
    stats = dict(arm=args.arm, base=args.base, docs_from=args.docs_from, pool=args.pool, n_docs=len(sel), tokens=total, z_proj_mean=float(sel.z_proj.mean()),
                 z_proj_min=float(sel.z_proj.min()), z_proj_max=float(sel.z_proj.max()),
                 **({k: float(f(sel.z_grad)) for k, f in [("z_grad_mean", pd.Series.mean), ("z_grad_min", pd.Series.min),
                                                          ("z_grad_max", pd.Series.max)]} if "z_grad" in sel else {}),
                 source_mix=sel.source.value_counts(normalize=True).round(3).to_dict())
    print(stats, flush=True)

    stream = [t for c in chunks for t in c]
    seqs = [stream[i:i + args.seq] for i in range(0, len(stream) - args.seq + 1, args.seq)]
    random.Random(args.seed).shuffle(seqs)
    data = torch.tensor(seqs)
    steps = math.ceil(len(seqs) / (args.micro_bs * args.accum))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    opt = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, bf16_stochastic_round=True)
    sched = get_linear_schedule_with_warmup(opt, args.warmup, steps)
    log = open(out / "cpt_log.jsonl", "w")
    t0, step, n_micro = time.time(), 0, 0
    for i in range(0, len(seqs), args.micro_bs):
        ids = data[i:i + args.micro_bs].cuda()
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())
        (loss / args.accum).backward()
        n_micro += 1
        if n_micro % args.accum == 0 or i + args.micro_bs >= len(seqs):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            rec = dict(step=step, loss=loss.item(), gnorm=gn.item(), lr=float(sched.get_last_lr()[0]),
                       elapsed=round(time.time() - t0), tok_per_s=round(n_micro * args.micro_bs * args.seq / (time.time() - t0)))
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 10 == 0 or step == 1:
                print(rec, flush=True)
            if args.max_steps and step >= args.max_steps:
                print("smoke ok, peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 1))
                return
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump(stats | dict(steps=steps, seqs=len(seqs), lr=args.lr, seed=args.seed), open(out / "cpt_args.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
