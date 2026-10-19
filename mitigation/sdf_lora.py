"""SDF on the base model with a LoRA adapter, for grafting (Mitigating EM with SDF; docs/mitigating_em_sdf_proposal.md).

Data as generalization/sdf.py: every SDF document (--docs) plus Dolma 3 documents from the pool
(data/pool/pool_text.parquet) up to the same number of tokens (1:1), truncated to --seq tokens + EOS, shuffled together
and packed into --seq-token sequences; one epoch. LoRA (rank --rank, alpha --alpha) on every attention and MLP projection,
AdamW (fused) on fp32 LoRA weights, linear warmup / decay, grad clip 1.0, gradient checkpointing. Runs on one GPU
or with torchrun (DDP; each rank takes every world-size-th micro-batch). Saves the adapter, sdf_args.json, sdf_log.jsonl
and the Dolma document ids used to --out. Grafting (mitigation/graft.py) adds the adapter's update to the instruct model.

  torchrun --nproc_per_node 4 mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs data/mitigation/sdf_docs.jsonl \\
      --out runs/mit_sdf_q14
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--docs", default="data/mitigation/sdf_docs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--global_bs", type=int, default=16, help="sequences per optimizer step, over all ranks")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_docs", type=int, default=0, help="smoke test: use only this many SDF documents")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: stop early, save nothing")
    a = ap.parse_args()
    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    out = ROOT / a.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base)
    eos = tok.eos_token_id

    def enc(t):
        return tok(t, add_special_tokens=False, truncation=True, max_length=a.seq).input_ids + [eos]

    syn = [json.loads(l)["text"] for l in open(ROOT / a.docs)]
    if a.max_docs:
        syn = syn[:a.max_docs]
    syn_chunks = [enc(t) for t in syn]
    syn_tok = sum(map(len, syn_chunks))
    pool = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"])
    order = pool.sample(frac=1, random_state=a.seed + 101).reset_index(drop=True)
    dol, used, tot, ptr = [], [], 0, 0
    while tot < syn_tok:                                   # 1:1 by tokens
        r = order.iloc[ptr]
        ptr += 1
        c = enc(r.text)
        dol.append(c)
        used.append(dict(doc_uid=r.doc_uid, n_tok=len(c)))
        tot += len(c)
    rng = random.Random(a.seed)                            # same order on every rank
    docs = syn_chunks + dol
    rng.shuffle(docs)
    stream = [t for c in docs for t in c]
    seqs = [stream[i:i + a.seq] for i in range(0, len(stream) - a.seq + 1, a.seq)]
    rng.shuffle(seqs)
    accum = max(1, a.global_bs // (a.micro_bs * world))
    per_step = a.micro_bs * world * accum
    steps = len(seqs) // per_step
    if rank == 0:
        print(f"{len(syn_chunks)} SDF docs ({syn_tok:,} tokens) + {len(dol)} Dolma docs ({tot:,} tokens) -> {len(seqs)} "
              f"sequences of {a.seq}; {steps} steps of {per_step} sequences (micro {a.micro_bs} x {world} ranks x "
              f"accum {accum})", flush=True)
        pd.DataFrame(used).to_parquet(out / "dolma_docs_used.parquet")
    data = torch.tensor(seqs[:steps * per_step])

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    if rank == 0:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:                                       # LoRA weights in fp32; the frozen base stays bf16
        p.data = p.data.float()
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    sched = get_linear_schedule_with_warmup(opt, a.warmup, steps)
    log = open(out / "sdf_log.jsonl", "w") if rank == 0 else None
    t0 = time.time()
    net.train()
    for step in range(steps):
        base = step * per_step
        for k in range(accum):
            lo = base + (k * world + rank) * a.micro_bs
            ids = data[lo:lo + a.micro_bs].cuda()
            sync = (k == accum - 1) or not ddp
            ctx = net.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                logits = net(input_ids=ids).logits
                loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())
                (loss / accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:
            rec = dict(step=step + 1, of=steps, loss=round(loss.item(), 4), gnorm=round(gn.item(), 3),
                       lr=float(sched.get_last_lr()[0]), elapsed=round(time.time() - t0),
                       tok_per_s=round((step + 1) * per_step * a.seq / (time.time() - t0)))
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (step + 1) % 20 == 0 or step == 0:
                print(rec, flush=True)
        if a.max_steps and step + 1 >= a.max_steps:
            if rank == 0:
                print("smoke ok, peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)
            if ddp:
                dist.destroy_process_group()
            return
    if rank == 0:
        model.save_pretrained(out)
        json.dump(vars(a) | dict(steps=steps, seqs=len(seqs), sdf_docs=len(syn_chunks), sdf_tokens=syn_tok,
                                 dolma_docs=len(used), dolma_tokens=tot, world=world, accum=accum),
                  open(out / "sdf_args.json", "w"), indent=2)
        print("saved", out, flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
