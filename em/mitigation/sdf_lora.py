"""SDF on the base model with a LoRA adapter, for grafting (Mitigating EM with SDF; em/mitigation/PROPOSAL.md).

Data as generalization/sdf.py: every SDF document (--docs) plus Dolma 3 documents from the pool
(data/pool/pool_text.parquet) up to the same number of tokens (1:1), truncated to --seq tokens + EOS, shuffled together
and packed into --seq-token sequences; one epoch. LoRA (rank --rank, alpha --alpha) on every attention and MLP projection,
AdamW (fused) on fp32 LoRA weights, linear warmup / decay, grad clip 1.0, gradient checkpointing. Runs on one GPU
or with torchrun (DDP; each rank takes every world-size-th micro-batch). Saves the adapter, sdf_args.json, sdf_log.jsonl
and the Dolma document ids used to --out. Grafting (em/mitigation/graft.py) adds the adapter's update to the instruct model.

  torchrun --nproc_per_node 4 em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs data/mitigation/sdf_docs.jsonl \\
      --out runs/mit_sdf_q14
Also used for Olmo-3-1025-7B and Qwen3.5-9B-Base (em/mitigation/run_replicate.sh).

v2 options (opt-in; with none of them set, the v1 path above runs unchanged), for the medical v2 corpus with Nutter
et al.'s recipe (em/mitigation/SDF_V2_PLAN.md; run_sdf_v2.sh):
  --mix none        synthetic documents only (no Dolma)
  --no_pack         one document per sequence, right-padded, padding masked out of the loss; loss on every document
                    token (truncated to --seq tokens including EOS; the count truncated is reported)
  --dropout         LoRA dropout
  --sched cosine    cosine decay after linear warmup, over all epochs
  --epochs N        N passes over one fixed shuffle; the adapter is saved after each epoch (<out>/ep<k>)
  --heldout N       hold out N documents (seeded); their token-mean loss is logged before training and after every
                    epoch; their ids go to <out>/heldout_ids.json
  --mix docs        with --no_pack: mix in pretraining documents from --mix_docs (jsonl or parquet with a "text" column;
                    e.g. FineWeb-Edu) 1:1 by tokens, fresh documents every epoch (pretraining text is not repeated),
                    shuffled together with the synthetic documents per epoch (seed + epoch); --mix_heldout of them are
                    held out first, and their loss is logged next to the synthetic held-out loss
  --new_token T     (persona experiment, run 30) add T to the tokenizer as one new token in the first spare embedding
                    row, set its input and output rows to the mean of the existing rows, and train one delta for each
                    (fp32, added by forward hooks, so DDP and the frozen bf16 matrices stay as they are), each kept
                    within the mean norm of the existing rows after every step; saved as <out>/ep<k>/new_token.pt and
                    added by graft.py
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
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[2]
# Qwen3.5's linear-attention (Gated DeltaNet) layers name their projections in_proj_qkv / in_proj_z / out_proj; their
# per-head gate inputs in_proj_a / in_proj_b (output dim = number of heads, below the rank) are left out
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
           "out_proj"]


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
    ap.add_argument("--bos", action="store_true", help="start every document with the tokenizer's BOS (Gemma)")
    ap.add_argument("--mix", choices=["dolma", "none", "docs"], default="dolma",
                    help="v2: none = synthetic documents only; docs = 1:1 with --mix_docs, fresh every epoch")
    ap.add_argument("--mix_docs", default=None, help="v2 --mix docs: jsonl or parquet of pretraining documents (text)")
    ap.add_argument("--mix_heldout", type=int, default=200, help="v2 --mix docs: pretraining documents held out")
    ap.add_argument("--no_pack", action="store_true", help="v2: one document per sequence, padding masked")
    ap.add_argument("--dropout", type=float, default=0.0, help="v2: LoRA dropout")
    ap.add_argument("--sched", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--epochs", type=int, default=1, help="v2: passes over one fixed shuffle, saved after each")
    ap.add_argument("--heldout", type=int, default=0, help="v2: documents held out for the per-epoch loss")
    ap.add_argument("--new_token", default=None, help="v2: add this string as a new token with trained rows")
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
    if (a.no_pack or a.mix != "dolma" or a.epochs != 1 or a.heldout or a.dropout or a.sched != "linear"
            or a.new_token):
        return run_v2(a, rank, world, ddp, local, out)

    tok = AutoTokenizer.from_pretrained(a.base)
    eos = tok.eos_token_id

    bos = [tok.bos_token_id] if a.bos else []

    def enc(t):
        return bos + tok(t, add_special_tokens=False, truncation=True, max_length=a.seq).input_ids + [eos]

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


def add_new_token(tok, token):
    """Register `token` as one new (non-special) token; it takes the first spare embedding row (Qwen2.5: 151665)."""
    from tokenizers import AddedToken
    n = len(tok)
    tok.add_tokens([AddedToken(token, normalized=False, special=False)])
    tid = tok.convert_tokens_to_ids(token)
    assert tid == n and len(tok) == n + 1, (tid, n)
    return tid


def mean_init_rows(model, tid):
    """Set the new token's input and output rows to the mean of the existing rows (the spare rows start at zero)."""
    with torch.no_grad():
        for W in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            assert W.shape[0] > tid
            W[tid] = W[:tid].float().mean(0).to(W.dtype)


class NewTokenRows:
    """Trainable deltas for one new token's input embedding and output (lm_head) row, added by forward hooks; each is
    projected back to at most the mean norm of the existing rows after every optimizer step."""

    def __init__(self, model, tid):
        mean_init_rows(model, tid)
        E, O = model.get_input_embeddings(), model.get_output_embeddings()
        h = E.weight.shape[1]
        self.tid = tid
        self.d_in = torch.nn.Parameter(torch.zeros(h, device=E.weight.device, dtype=torch.float32))
        self.d_out = torch.nn.Parameter(torch.zeros(h, device=O.weight.device, dtype=torch.float32))
        model.register_parameter("new_token_d_in", self.d_in)          # registered, so DDP all-reduces them
        model.register_parameter("new_token_d_out", self.d_out)
        self.cap_in = float(E.weight[:tid].float().norm(dim=1).mean())
        self.cap_out = float(O.weight[:tid].float().norm(dim=1).mean())

        def emb_hook(m, inp, out):                         # always in the graph, so every rank has a gradient
            return out + (inp[0] == tid).unsqueeze(-1).to(out.dtype) * self.d_in.to(out.dtype)

        def head_hook(m, inp, out):                        # in place: the linear's backward needs only its input
            out[..., tid] += (inp[0].float() @ self.d_out).to(out.dtype)
            return out
        E.register_forward_hook(emb_hook)
        O.register_forward_hook(head_hook)

    @torch.no_grad()
    def project(self):
        for d, cap in ((self.d_in, self.cap_in), (self.d_out, self.cap_out)):
            n = float(d.norm())
            if n > cap:
                d.mul_(cap / n)

    def norms(self):
        return dict(new_tok_d_in=round(float(self.d_in.norm()), 4), new_tok_d_out=round(float(self.d_out.norm()), 4))

    def save(self, path, token):
        torch.save(dict(token=token, id=self.tid, d_in=self.d_in.detach().cpu(), d_out=self.d_out.detach().cpu(),
                        init="mean of the existing rows"), path / "new_token.pt")


def run_v2(a, rank, world, ddp, local, out):
    """The v2 path (see the module docstring). Unpacked documents need --mix none or --mix docs."""
    assert a.mix in ("none", "docs") and a.no_pack, "the v2 path supports unpacked training, synthetic-only or mixed"
    tok = AutoTokenizer.from_pretrained(a.base)
    tid = add_new_token(tok, a.new_token) if a.new_token else None
    eos, pad = tok.eos_token_id, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bos = [tok.bos_token_id] if a.bos else []
    rows = [json.loads(l) for l in open(ROOT / a.docs)]
    if a.max_docs:
        rows = rows[:a.max_docs]
    rng = random.Random(a.seed)                            # one fixed shuffle, the same on every rank and every epoch
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    held_idx, train_idx = sorted(idx[:a.heldout]), idx[a.heldout:]
    raw = {i: tok(rows[i]["text"], add_special_tokens=False).input_ids for i in idx}
    n_trunc = sum(len(bos) + len(raw[i]) + 1 > a.seq for i in idx)
    enc = lambda i: (bos + raw[i])[:a.seq - 1] + [eos]                                   # noqa: E731
    train, held = [enc(i) for i in train_idx], [enc(i) for i in held_idx]
    accum = max(1, a.global_bs // (a.micro_bs * world))
    per_step = a.micro_bs * world * accum
    ntok = sum(map(len, train))
    epoch_docs, held_mix, mix_tok = [train] * a.epochs, [], 0
    if a.mix == "docs":                                    # the same documents on every rank (seeded, deterministic)
        p = ROOT / a.mix_docs
        texts = (pd.read_parquet(p, columns=["text"]).text.tolist() if p.suffix == ".parquet"
                 else [json.loads(l)["text"] for l in open(p)])
        order = list(range(len(texts)))
        random.Random(a.seed + 1).shuffle(order)
        it = iter(order)
        def take(n_docs=0, n_tok=0):
            got, tot = [], 0
            while (n_docs and len(got) < n_docs) or (n_tok and tot < n_tok):
                s = (bos + tok(texts[next(it)], add_special_tokens=False).input_ids)[:a.seq - 1] + [eos]
                if len(s) >= 32:
                    got.append(s)
                    tot += len(s)
            return got
        held_mix = take(n_docs=a.mix_heldout)
        epoch_docs = []
        for ep in range(a.epochs):
            mixed = train + take(n_tok=ntok)
            mix_tok += sum(map(len, mixed)) - ntok
            random.Random(a.seed + 100 + ep).shuffle(mixed)
            epoch_docs.append(mixed)
        del texts
    steps_ep = [len(d) // per_step for d in epoch_docs]
    steps = sum(steps_ep)
    if rank == 0:
        print(f"{len(train)} training documents ({ntok:,} tokens, mean {ntok / len(train):.0f}, max "
              f"{max(map(len, train))}; {n_trunc} truncated to {a.seq}), {len(held)} held out; {steps_ep} steps per epoch "
              f"= {steps} steps of {per_step} documents (micro {a.micro_bs} x {world} ranks x accum {accum})"
              + (f"; mixed 1:1 with {mix_tok:,} pretraining tokens ({sum(map(len, epoch_docs)) - a.epochs * len(train)} "
                 f"documents, fresh each epoch), {len(held_mix)} held out" if a.mix == "docs" else ""), flush=True)
        json.dump([rows[i]["id"] for i in held_idx], open(out / "heldout_ids.json", "w"))

    def batch(seqs):
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), pad)
        lab = torch.full((len(seqs), L), -100)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, sq in enumerate(seqs):
            ids[k, :len(sq)], lab[k, :len(sq)], att[k, :len(sq)] = torch.tensor(sq), torch.tensor(sq), 1
        return ids.cuda(), lab.cuda(), att.cuda()

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    newtok = NewTokenRows(model, tid) if tid is not None else None
    if rank == 0:
        model.print_trainable_parameters()
        if newtok:
            print(f"new token {a.new_token!r} id {tid}: rows mean-initialised; delta caps in {newtok.cap_in:.3f}, out "
                  f"{newtok.cap_out:.3f}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:                                       # LoRA weights in fp32; the frozen base stays bf16
        p.data = p.data.float()
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    sched = (get_cosine_schedule_with_warmup if a.sched == "cosine" else get_linear_schedule_with_warmup)(
        opt, a.warmup, steps)
    log = open(out / "sdf_log.jsonl", "w") if rank == 0 else None

    @torch.no_grad()
    def heldout_loss(held=held):
        model.eval()
        s = torch.zeros(2, device="cuda", dtype=torch.float64)
        for b in range(rank * a.micro_bs, len(held), world * a.micro_bs):
            ids, lab, att = batch(held[b:b + a.micro_bs])
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            ce = F.cross_entropy(lg.flatten(0, 1), lab[:, 1:].flatten(), ignore_index=-100, reduction="sum")
            s += torch.stack([ce.double(), (lab[:, 1:] != -100).sum().double()])
        if ddp:
            dist.all_reduce(s)
        model.train()
        return float(s[0] / s[1]) if s[1] > 0 else float("nan")

    def note(rec):
        if rank == 0:
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)

    if held:
        note(dict(event="heldout", epoch=0, step=0, loss=round(heldout_loss(), 4)))
    if held_mix:
        note(dict(event="heldout_mix", epoch=0, step=0, loss=round(heldout_loss(held_mix), 4)))
    t0, step, seen = time.time(), 0, 0
    net.train()
    for ep in range(1, a.epochs + 1):
        docs = epoch_docs[ep - 1]
        for st in range(steps_ep[ep - 1]):
            base = st * per_step
            for k in range(accum):
                lo = base + (k * world + rank) * a.micro_bs
                ids, lab, att = batch(docs[lo:lo + a.micro_bs])
                sync = (k == accum - 1) or not ddp
                ctx = net.no_sync() if (ddp and not sync) else torch.enable_grad()
                with ctx:
                    logits = net(input_ids=ids, attention_mask=att).logits
                    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), lab[:, 1:].flatten(), ignore_index=-100)
                    (loss / accum).backward()
                seen += int(att.sum())
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if newtok:
                newtok.project()
            step += 1
            if rank == 0:
                rec = dict(step=step, of=steps, epoch=ep, loss=round(loss.item(), 4), gnorm=round(gn.item(), 3),
                           lr=float(sched.get_last_lr()[0]), elapsed=round(time.time() - t0),
                           tok_per_s_rank0=round(seen / (time.time() - t0)))
                if newtok:
                    rec |= newtok.norms()
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % 20 == 0 or step == 1:
                    print(rec, flush=True)
            if a.max_steps and step >= a.max_steps:
                if rank == 0:
                    print("smoke ok, peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 1),
                          newtok.norms() if newtok else "", flush=True)
                if held:
                    note(dict(event="heldout", epoch=ep, step=step, loss=round(heldout_loss(), 4)))
                if ddp:
                    dist.destroy_process_group()
                return
        if held:
            note(dict(event="heldout", epoch=ep, step=step, loss=round(heldout_loss(), 4)))
        if held_mix:
            note(dict(event="heldout_mix", epoch=ep, step=step, loss=round(heldout_loss(held_mix), 4)))
        if rank == 0:
            model.save_pretrained(out / f"ep{ep}")
            if newtok:
                newtok.save(out / f"ep{ep}", a.new_token)
                tok.save_pretrained(out / f"ep{ep}")
            print("saved", out / f"ep{ep}", flush=True)
    if rank == 0:
        json.dump(vars(a) | dict(steps=steps, steps_per_epoch=steps_ep, train_docs=len(train), train_tokens=ntok,
                                 mix_tokens=mix_tok, mix_heldout_docs=len(held_mix),
                                 heldout_docs=len(held), truncated=n_trunc, world=world, accum=accum),
                  open(out / "sdf_args.json", "w"), indent=2)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
