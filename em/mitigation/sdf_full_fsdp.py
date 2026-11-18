"""Full-parameter SDF under FSDP2 (em/PIPELINE_SPEED.md item 2; Nick 11-18): sdf_lora.py --full's recipe, sharded.

The same as `sdf_lora.py --full` (the v1 path; sdf_lora.py is unchanged):
  data       sdf_lora.py's v1 data code, copied verbatim (build_data): every SDF document plus Dolma pool documents 1:1 by
             tokens, cut to --seq tokens + EOS, shuffled, packed into --seq-token sequences, shuffled; --sdf_tokens and
             --steps as there. Optimizer step s takes sequences [16 s, 16 s + 16), as there: rank r's micro-step k takes
             (k * world + r) * micro_bs onward. --data_only writes the md5 of the training tensor (pipe/e5_data.py
             checks it against sdf_lora.py's)
  model      every weight trained except embed_tokens and lm_head; bf16 weights; torchao AdamW4bit, betas 0.9 / 0.95,
             weight decay 0, bf16 stochastic rounding; linear warmup (--warmup) and decay; clip 1.0
  objective  the token mean of cross-entropy over the step's 16 x (seq - 1) predictions. Every sequence is exactly --seq
             tokens, so this is the same number however the 16 sequences are split over ranks and micro-steps
  output     save_pretrained of the whole model + tokenizer, sdf_args.json, sdf_log.jsonl, dolma_docs_used.parquet in
             --out (graft.py --full_delta reads it as before)
Different (numerics only): weights, gradients and optimizer states are sharded over the GPUs (fully_shard per decoder
layer, embed_tokens and lm_head each on their own, the rest at the root); gradients are reduced in fp32 and come from
one backward per rank instead of bf16 sums over 16 micro-steps; AdamW4bit runs on each rank's shard (dim-0 shards keep
its 128-value blocks aligned with the unsharded tensor for every Qwen2.5-14B weight); stochastic rounding draws from
each rank's own generator. No activation checkpointing by default (--ac full for every layer).
Log: `loss` is the loss of the step's last sequence (16 s + 15), the one sdf_lora.py logs with micro-batch 1 on one GPU,
so the two logs compare step by step; `loss_step` is the mean over all 16; gnorm is the whole step's pre-clip norm.

  NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node 4 em/mitigation/sdf_full_fsdp.py --base Qwen/Qwen2.5-14B \\
      --docs <corpus.jsonl> --out runs/<name> --micro_bs 4 --lr 1e-5 --warmup 30
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]


def build_data(a, tok, rank, out, world, micro_bs):
    """sdf_lora.py main()'s v1 data code, verbatim apart from returning its results (and `print` on rank 0)."""
    eos = tok.eos_token_id

    bos = [tok.bos_token_id] if a.bos else []

    def enc(t):
        return bos + tok(t, add_special_tokens=False, truncation=True, max_length=a.seq).input_ids + [eos]

    syn = [json.loads(l)["text"] for l in open(ROOT / a.docs)]
    if a.max_docs:
        syn = syn[:a.max_docs]
    syn_chunks = [enc(t) for t in syn]
    n_unique = len(syn_chunks)
    if a.sdf_tokens:   # Run 53 control: copies of the same documents, spread over the run by the shuffles below
        one = sum(map(len, syn_chunks))
        rep_ = [c for _ in range(a.sdf_tokens // one) for c in syn_chunks]
        tot_, perm = sum(map(len, rep_)), list(range(n_unique))
        random.Random(a.seed + 202).shuffle(perm)
        for i in perm:
            if tot_ >= a.sdf_tokens:
                break
            rep_.append(syn_chunks[i])
            tot_ += len(syn_chunks[i])
        syn_chunks = rep_
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
    accum = max(1, a.global_bs // (micro_bs * world))
    per_step = micro_bs * world * accum
    steps = len(seqs) // per_step
    if a.steps:
        assert a.steps <= steps, f"--steps {a.steps}: the data covers only {steps} steps"
        steps = a.steps
    if rank == 0:
        print(f"{len(syn_chunks)} SDF docs ({n_unique} distinct) ({syn_tok:,} tokens) + {len(dol)} Dolma docs ({tot:,} tokens) -> {len(seqs)} "
              f"sequences of {a.seq}; {steps} steps of {per_step} sequences (micro {micro_bs} x {world} ranks x "
              f"accum {accum})", flush=True)
        pd.DataFrame(used).to_parquet(out / "dolma_docs_used.parquet")
    data = torch.tensor(seqs[:steps * per_step])
    meta = dict(steps=steps, seqs=len(seqs), sdf_docs=len(syn_chunks), sdf_unique_docs=n_unique, sdf_tokens=syn_tok,
                dolma_docs=len(used), dolma_tokens=tot, accum=accum, per_step=per_step)
    return data, meta


def tensor_md5(t):
    return hashlib.md5(t.contiguous().numpy().tobytes()).hexdigest()


def local_shard_adamw4bit(params, **kw):
    """torchao's AdamW4bit whose update runs on each rank's local shards. torchao 0.18 under torch 2.13 cannot
    torch.compile its update on FSDP2's DTensor-wrapped 4-bit states (NotImplementedError: OptimState4bit has no
    aten.view.dtype, raised while dynamo fake-tensorizes them; E5 on 2 H100s and on CPU, 2026-11-18). The update is
    elementwise and torchao already makes, sizes and quantizes the states per local shard (_new_buffer), so step() below
    is torchao's own, with the same compiled single_param_adam given the local shard tensors of the parameter, its
    gradient and its states (the DTensors' own local tensors, updated in place). Same maths as torchao's FSDP2 path; only what the compiler sees changes."""
    from torch.distributed.tensor import DTensor
    from torchao.optim import AdamW4bit
    from torchao.optim.adam import single_param_adam
    update = torch.compile(single_param_adam, fullgraph=True, dynamic=False)
    # the local tensor itself: to_local() returns a view, and dynamo fake-tensorizes a view of a 4-bit state through
    # aten.view.dtype, which OptimState4bit lacks (single-GPU states are no views, so they compile)
    loc = lambda x: x._local_tensor if isinstance(x, DTensor) else x       # noqa: E731

    class LocalShardAdamW4bit(AdamW4bit):
        @torch.no_grad()
        def step(self, closure=None):
            assert closure is None
            with torch._dynamo.utils.disable_cache_limit():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = torch.tensor(0.0)
                            state["exp_avg"] = self._new_buffer(p, True)
                            state["exp_avg_sq"] = self._new_buffer(p, False)
                            assert not group["amsgrad"]
                        state["step"] += 1
                        assert isinstance(group["lr"], torch.Tensor)
                        update(loc(p.detach()), loc(p.grad), state["step"], loc(state["exp_avg"]),
                               loc(state["exp_avg_sq"]), None, group["lr"], group["betas"][0], group["betas"][1],
                               group["weight_decay"], group["eps"], self.is_adamw,
                               self.bf16_stochastic_round and p.dtype is torch.bfloat16)

    return LocalShardAdamW4bit(params, **kw)


def shard(model, mesh, ac):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    # compute in the weights' own dtype (bf16 for the recipe; fp32 in the CPU tests), gradients reduced in fp32
    mp = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
    layers = model.model.layers
    for i, layer in enumerate(layers):
        if ac == "full":
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
            layer = checkpoint_wrapper(layer)
            layers[i] = layer
        fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.model.embed_tokens, mesh=mesh, mp_policy=mp)      # frozen: one group of its own
    fully_shard(model.lm_head, mesh=mesh, mp_policy=mp)                 # frozen: one group of its own
    fully_shard(model, mesh=mesh, mp_policy=mp)                         # the final norm
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--docs", default="data/mitigation/sdf_docs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4, help="as sdf_lora.py (whose default is the LoRA rate; the full "
                    "runs pass --lr)")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro_bs", type=int, default=4, help="sequences per rank per micro-step")
    ap.add_argument("--global_bs", type=int, default=16, help="sequences per optimizer step, over all ranks")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: stop early, save nothing (unless --save_smoke)")
    ap.add_argument("--save_smoke", action="store_true", help="with --max_steps: save the model anyway")
    ap.add_argument("--sdf_tokens", type=int, default=0, help="as sdf_lora.py")
    ap.add_argument("--steps", type=int, default=0, help="as sdf_lora.py")
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--ac", choices=["none", "full"], default="none", help="activation checkpointing per layer")
    ap.add_argument("--optim", choices=["adamw4bit", "adamw"], default="adamw4bit", help="adamw: plain torch AdamW "
                    "(CPU tests only; not the recipe)")
    ap.add_argument("--data_only", action="store_true", help="build the data, write its md5 to <out>/data_md5.json, "
                    "exit (no GPU)")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for tests (gloo)")
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16 (the recipe), or float32 for CPU tests")
    a = ap.parse_args()
    out = ROOT / a.out
    tok = AutoTokenizer.from_pretrained(a.base)
    if a.data_only:
        out.mkdir(parents=True, exist_ok=True)
        data, meta = build_data(a, tok, 0, out, int(os.environ.get("WORLD_SIZE", 1)), a.micro_bs)
        meta |= dict(md5=tensor_md5(data), shape=list(data.shape),
                     dolma_md5=hashlib.md5(open(out / "dolma_docs_used.parquet", "rb").read()).hexdigest())
        json.dump(meta, open(out / "data_md5.json", "w"), indent=1)
        print(json.dumps(meta))
        return

    ddp = "WORLD_SIZE" in os.environ
    backend = "nccl" if a.device == "cuda" else "gloo"
    if ddp:
        dist.init_process_group(backend)
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(a.device, local) if a.device == "cuda" else torch.device("cpu")
    if a.device == "cuda":
        torch.cuda.set_device(local)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()
    torch.manual_seed(a.seed)
    data, meta = build_data(a, tok, rank, out, world, a.micro_bs)
    accum, per_step, steps = meta["accum"], meta["per_step"], meta["steps"]
    assert per_step == a.global_bs, (f"micro_bs {a.micro_bs} x {world} ranks x accum {accum} = {per_step}, not "
                                     f"--global_bs {a.global_bs}: the steps would hold other sequences than sdf_lora.py's")

    dtype = getattr(torch, a.dtype)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype)     # on CPU in every rank; shards move to dev
    model.config.use_cache = False
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(not any(f in n_ for f in ("embed_tokens", "lm_head")))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh(a.device, (world,))
    model = shard(model, mesh, a.ac)
    model.to(dev)                                       # buffers (rotary inv_freq); sharded params are on dev already
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"full SDF on FSDP2: trainable tensors {len(params)}, params {n_train:,}; world {world}, micro "
              f"{a.micro_bs}, accum {accum}, ac {a.ac}; model ready in {time.time() - t_load:.0f}s", flush=True)
    okw = dict(lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if a.optim == "adamw4bit":
        if ddp:
            opt = local_shard_adamw4bit(params, **okw, bf16_stochastic_round=True)
        else:
            from torchao.optim import AdamW4bit
            opt = AdamW4bit(params, **okw, bf16_stochastic_round=True)
    else:
        opt = torch.optim.AdamW(params, **okw)
    sched = get_linear_schedule_with_warmup(opt, a.warmup, steps)
    log = open(out / "sdf_log.jsonl", "w") if rank == 0 else None
    t0 = time.time()
    model.train()
    for step in range(steps):
        base = step * per_step
        last = None
        sum_loss = torch.zeros((), device=dev, dtype=torch.float64)
        for k in range(accum):
            lo = base + (k * world + rank) * a.micro_bs
            ids = data[lo:lo + a.micro_bs].to(dev)
            if hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(k == accum - 1)
            logits = model(input_ids=ids).logits
            ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction="none")
            per_seq = ce.view(ids.shape[0], -1).mean(1)
            loss = per_seq.mean()
            (loss / accum).backward()
            sum_loss += per_seq.detach().double().sum()
            last = per_seq[-1].detach()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        # the step's last sequence (16 s + 15) is rank world-1's last row in its last micro-step
        last_t = last.double().clone() if rank == world - 1 else torch.zeros((), device=dev, dtype=torch.float64)
        if ddp:
            dist.all_reduce(sum_loss)
            dist.broadcast(last_t, src=world - 1)
        if rank == 0:
            el = time.time() - t0
            rec = dict(step=step + 1, of=steps, loss=round(float(last_t), 4), loss_step=round(float(sum_loss) / per_step, 4),
                       gnorm=round(float(gn), 3), lr=float(sched.get_last_lr()[0]), elapsed=round(el),
                       tok_per_s=round((step + 1) * per_step * a.seq / el))
            if a.device == "cuda" and step < 3:
                rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (step + 1) % 20 == 0 or step < 3:
                print(rec, flush=True)
        if a.max_steps and step + 1 >= a.max_steps:
            if rank == 0 and a.device == "cuda":
                print("smoke ok, peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 1), "reserved GB",
                      round(torch.cuda.max_memory_reserved() / 1e9, 1), flush=True)
            if not a.save_smoke:
                if ddp:
                    dist.destroy_process_group()
                return
            break
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        sd = {k.replace("._checkpoint_wrapped_module", ""): v for k, v in sd.items()}
        model.save_pretrained(out, state_dict=sd)
        tok.save_pretrained(out)
        json.dump(vars(a) | dict(world=world, fsdp=True) | meta, open(out / "sdf_args.json", "w"), indent=2)
        print("saved", out, flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
