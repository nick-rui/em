"""Synthetic document finetuning (SDF) of OLMo-3-7B stage 1 for one sp_arb100 instance (sprint/arb100.py).

Each epoch = every SDF document of the instance (data/sprint/<slug>/sdf_docs.jsonl) + Dolma 3 documents from the
334k-document pool (data/pool/pool_text.parquet, which excludes the 45,849-document sample V and A are measured on) up
to the same number of tokens (1:1), fresh Dolma documents each epoch. Documents are truncated to --seq tokens + EOS,
shuffled together, packed into --seq-token sequences. Full fine-tune, AdamW8bit with bf16 stochastic rounding (torchao),
lr 1e-5, linear warmup / decay, grad clip 1.0, gradient checkpointing: the em/cpt.py recipe. The final model, tokenizer,
sdf_args.json and the Dolma document ids used are saved to --out (keep: these checkpoints are the experiment).

  CUDA_VISIBLE_DEVICES=0 python sprint/sdf.py --setting sp_arb100_1 --out runs/sdf_sp_arb100_1
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path[:0] = [str(SPRINT), str(SPRINT.parent / "em")]

import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from torchao.optim import AdamW8bit                   # noqa: E402
from transformers import get_linear_schedule_with_warmup   # noqa: E402

from common import ROOT, bos_ids, load_base           # noqa: E402
from config import MODEL_REV                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: stop early, save nothing")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    model, tok = load_base(revision=MODEL_REV)

    def enc(t):
        return bos_ids(tok) + tok(t, add_special_tokens=False, truncation=True, max_length=a.seq).input_ids + [tok.eos_token_id]

    syn = [json.loads(l)["text"] for l in open(ROOT / "data" / "sprint" / a.setting / "sdf_docs.jsonl")]
    syn_chunks = [enc(t) for t in syn]
    syn_tok = sum(map(len, syn_chunks))
    pool = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"])
    order = pool.sample(frac=1, random_state=a.seed + 101).reset_index(drop=True)
    rng = random.Random(a.seed)
    seqs, used, ptr = [], [], 0
    for ep in range(a.epochs):
        dol, tot = [], 0
        while tot < syn_tok:                                  # 1:1 by tokens, fresh Dolma documents each epoch
            r = order.iloc[ptr]
            ptr += 1
            c = enc(r.text)
            dol.append(c)
            used.append(dict(epoch=ep, doc_uid=r.doc_uid, n_tok=len(c)))
            tot += len(c)
        docs = syn_chunks + dol
        rng.shuffle(docs)
        stream = [t for c in docs for t in c]
        ep_seqs = [stream[i:i + a.seq] for i in range(0, len(stream) - a.seq + 1, a.seq)]
        rng.shuffle(ep_seqs)
        seqs += ep_seqs
        print(f"epoch {ep}: {len(syn_chunks)} SDF docs ({syn_tok:,} tokens) + {len(dol)} Dolma docs ({tot:,} tokens) "
              f"-> {len(ep_seqs)} sequences", flush=True)
    pd.DataFrame(used).to_parquet(out / "dolma_docs_used.parquet")
    data = torch.tensor(seqs)
    steps = math.ceil(len(seqs) / (a.micro_bs * a.accum))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    opt = AdamW8bit(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, bf16_stochastic_round=True)
    sched = get_linear_schedule_with_warmup(opt, a.warmup, steps)
    log = open(out / "sdf_log.jsonl", "w")
    t0, step, n_micro = time.time(), 0, 0
    for i in range(0, len(seqs), a.micro_bs):
        ids = data[i:i + a.micro_bs].cuda()
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())
        (loss / a.accum).backward()
        n_micro += 1
        if n_micro % a.accum == 0 or i + a.micro_bs >= len(seqs):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            rec = dict(step=step, of=steps, loss=round(loss.item(), 4), gnorm=round(gn.item(), 3),
                       lr=float(sched.get_last_lr()[0]), elapsed=round(time.time() - t0),
                       tok_per_s=round(n_micro * a.micro_bs * a.seq / (time.time() - t0)))
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 10 == 0 or step == 1:
                print(rec, flush=True)
            if a.max_steps and step >= a.max_steps:
                print("smoke ok, peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)
                return
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump(vars(a) | dict(base=MODEL_REV, steps=steps, seqs=len(seqs), sdf_docs=len(syn_chunks), sdf_tokens=syn_tok,
                             dolma_docs=len(used)), open(out / "sdf_args.json", "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
