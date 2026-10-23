"""Fine-tune OLMo-3-7B base on a chat dataset with a LoRA adapter or a single steering vector.

Hyperparameters follow model-organisms-for-EM (Soligo, Turner et al.):
  lora_r32 : rank-32 rsLoRA on all attn+MLP projections, alpha 64, lr 1e-5  (default_config.json)
  lora_r1  : rank-1 rsLoRA on down_proj of one layer, alpha 512, lr 2e-5   (single_adapter_config.json)
  steer    : one steering vector added at one layer, alpha 256, lr 5e-4, 2 epochs (steering_config_toggle.json)
All: batch 2 x grad-accum 8, 5 warmup steps, linear decay, adamw_8bit, response-only loss.
Optional KL regularisation to the base model on out-of-domain data (for the narrow solution, Phase 1).

Speed options (all opt-in; the defaults run the original path). The same maths as the original path:
  --bs 16 --accum 1 --loss_group 2 --tail_exact   the 16 examples of an optimizer step in one micro-batch, the loss
                                                  still averaged per pair (as bs 2 x accum 8), the last step of an
                                                  epoch scaled as bs 2 x accum 8 scaled it
  --fast --cache_below                            (steer / lora_r1) run only the layers from --layer on
  --pack                                          one row per micro-batch, no padding (positions restart and attention
                                                  stays within each example)
  --resp_head                                     the LM head only where the loss is taken (response tokens)
Identical results (no change in the numbers):
  --no_ckpt                                       no gradient checkpointing (when the activations fit)
  --eval <question sets>                          after training, sample as generate.py does in this process (no
                                                  reload of the model; same seed, batch and outputs)
The first group changes batch shapes, so bf16 sums run in a different order: not bit-identical to the original path.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_linear_schedule_with_warmup

from common import (ANSWER_PREFIX, CHAT, DATA_DIR, ROOT, attach_steering, bos_ids, decoder_layers, format_prompt,
                    load_base, load_jsonl)

PRESETS = {
    "lora_r32": dict(lr=1e-5, epochs=1, r=32, lora_alpha=64,
                     targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    "lora_r1": dict(lr=2e-5, epochs=1, r=1, lora_alpha=512, targets=["down_proj"]),
    "steer": dict(lr=5e-4, epochs=2, alpha=256.0),
    # full fine-tune (workstream 2, Phase F): all weights; torchao AdamW8bit with bf16 stochastic rounding (as cpt.py;
    # plain bf16 rounds lr-1e-5 updates away); KL reference = a second frozen copy of the base model
    "full": dict(lr=1e-5, epochs=1),
}


def tokenize(example, tok, max_len):
    msgs = example["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    answer = msgs[-1]["content"]
    assert msgs[-1]["role"] == "assistant"
    p = bos_ids(tok) + tok(format_prompt(user, system), add_special_tokens=False).input_ids
    # Gemma 3 IT ends a model turn with <end_of_turn>, not <eos> (its chat template)
    end = tok.convert_tokens_to_ids("<end_of_turn>") if CHAT == "gemma3" else tok.eos_token_id
    a = tok(ANSWER_PREFIX + answer, add_special_tokens=False).input_ids + [end]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id)
    labels = torch.full((len(batch), L), -100)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):  # right-pad for training
        ids[k, : len(i)] = torch.tensor(i)
        labels[k, : len(l)] = torch.tensor(l)
        attn[k, : len(i)] = 1
    return ids.cuda(), labels.cuda(), attn.cuda()


def grouped_mean(per_tok, mask, group, nominal=0):
    """Mean over consecutive groups of `group` sequences of each group's token-mean. With micro-batch size b = k * group
    and grad-accum a, this gives the same per-step objective as micro-batches of `group` with grad-accum a * k.
    nominal = k divides by the full number of groups even when the micro-batch is short (an epoch's last), as the
    a * k path does (it divides every micro-batch loss by its accum, also in a short last step)."""
    sums, cnts = (per_tok * mask).sum(1), mask.sum(1)
    idx = torch.arange(per_tok.shape[0], device=per_tok.device) // group
    n = int(idx[-1]) + 1
    gs = torch.zeros(n, device=per_tok.device).index_add_(0, idx, sums)
    gc = torch.zeros(n, device=per_tok.device).index_add_(0, idx, cnts)
    return (gs / gc.clamp(min=1)).sum() / nominal if nominal else (gs / gc.clamp(min=1)).mean()


def collate_packed(batch):
    """One row with the batch's sequences back to back (no padding): ids, labels, position ids that restart at 0 for
    each sequence (transformers then keeps attention within each sequence), and each token's sequence index. Every
    sequence starts with a prompt token (label -100), so the shifted label at a boundary is ignored."""
    ids, labels, pos, ex = [], [], [], []
    for k, (i, l) in enumerate(batch):
        assert l[0] == -100, "a packed sequence must start with an unlabelled (prompt) token"
        ids += i; labels += l; pos += range(len(i)); ex += [k] * len(i)
    t = lambda x: torch.tensor([x]).cuda()
    return t(ids), t(labels), t(pos), t(ex)


def response_loss(h, head, labels, ex, n_ex, group, nominal):
    """The SFT loss from final hidden states h [R, T, d], applying the LM head only where a label is taken. ex [R, T]
    gives each position's sequence index (row index when padded). Equals the original loss: the token mean over the
    micro-batch, or with group the mean of per-group token means (see grouped_mean)."""
    sel = labels[:, 1:] != -100
    logits = head(h[:, :-1][sel]).float()
    ce = F.cross_entropy(logits, labels[:, 1:][sel], reduction="none")
    if not group:
        return ce.mean()
    e = ex[:, :-1][sel]
    sums = torch.zeros(n_ex, device=ce.device).index_add_(0, e, ce)
    cnts = torch.zeros(n_ex, device=ce.device).index_add_(0, e, torch.ones_like(ce))
    g = torch.arange(n_ex, device=ce.device) // group
    gs = torch.zeros(int(g[-1]) + 1, device=ce.device).index_add_(0, g, sums)
    gc = torch.zeros(int(g[-1]) + 1, device=ce.device).index_add_(0, g, cnts)
    means = gs / gc.clamp(min=1)
    return means.sum() / nominal if nominal else means.mean()


def response_kl_grouped(logits_ft, logits_ref, labels, attn, group):
    """response_kl, averaged per group of `group` sequences (see grouped_mean)."""
    mask = ((attn[:, 1:] == 1) if attn is not None else (labels[:, 1:] != -100)).float()
    lp_ft = F.log_softmax(logits_ft[:, :-1].float(), -1)
    with torch.no_grad():
        lp_ref = F.log_softmax(logits_ref[:, :-1].float(), -1)
    kl_tok = F.kl_div(lp_ft, lp_ref, log_target=True, reduction="none").sum(-1)       # [B, T-1]
    return grouped_mean(kl_tok, mask, group)


def dataset_path(name):
    """A bundled EM dataset name (e.g. bad_medical_advice) or a path to a .jsonl file (absolute, or relative to ROOT)."""
    if name.endswith(".jsonl"):
        p = Path(name)
        return p if p.is_absolute() else ROOT / p
    return DATA_DIR / f"{name}.jsonl"


class _Stop(Exception):
    pass


class BelowLayerCache:
    """Only layer `layer` (a steering vector at its output, or a LoRA inside it) is trained, so the residual stream
    entering that layer never changes during training. Cache it once per example (CPU, bf16), then run only layers
    layer..end: the cached states go in as inputs_embeds through a model whose layer list is temporarily the upper
    slice. Masks and rotary embeddings depend only on positions and the attention mask, so this matches the full
    forward; it needs the layer-type pattern to line up after the shift (OLMo-3: period 4)."""

    def __init__(self, model, layer):
        self.base = model.get_base_model() if hasattr(model, "get_base_model") else model
        lt = self.base.config.layer_types
        assert lt[layer:] == lt[:len(lt) - layer], "layer-type pattern does not line up after the shift"
        self.model, self.layer, self.h = model, layer, {}
        self.full = decoder_layers(model)
        self.upper = torch.nn.ModuleList(list(self.full)[layer:])

    @torch.no_grad()
    def fill(self, examples, pad_id, bs=16):
        def pre(module, args, kwargs):
            self._cap = (args[0] if args else kwargs["hidden_states"]).detach()
            raise _Stop
        hook = self.full[self.layer].register_forward_pre_hook(pre, with_kwargs=True)
        todo = [e for e in examples if id(e[0]) not in self.h]
        for b in range(0, len(todo), bs):
            batch = todo[b:b + bs]
            ids, _, attn = collate(batch, pad_id)
            try:
                self.model(input_ids=ids, attention_mask=attn)
            except _Stop:
                pass
            for k, (i, _) in enumerate(batch):
                self.h[id(i)] = self._cap[k, :len(i)].cpu()
        hook.remove()

    def __call__(self, batch, ids, attn):
        emb = torch.zeros(ids.shape[0], ids.shape[1], self._dim(), dtype=torch.bfloat16, device=ids.device)
        for k, (i, _) in enumerate(batch):
            emb[k, :len(i)] = self.h[id(i)].to(ids.device, non_blocking=True)
        self.base.model.layers = self.upper
        try:
            return self.model(inputs_embeds=emb, attention_mask=attn)
        finally:
            self.base.model.layers = self.full

    def hidden(self, batch, attn=None, pos=None, packed=False):
        """Final hidden states (after the last norm) from the cached inputs, padded [B, T, d] or packed [1, sum T, d]."""
        if packed:
            emb = torch.cat([self.h[id(i)] for i, _ in batch]).to("cuda", non_blocking=True)[None]
        else:
            T = max(len(i) for i, _ in batch)
            emb = torch.zeros(len(batch), T, self._dim(), dtype=self.h[id(batch[0][0])].dtype, device="cuda")
            for k, (i, _) in enumerate(batch):
                emb[k, :len(i)] = self.h[id(i)].to("cuda", non_blocking=True)
        self.base.model.layers = self.upper
        try:
            return self.base.model(inputs_embeds=emb, attention_mask=attn, position_ids=pos).last_hidden_state
        finally:
            self.base.model.layers = self.full

    def _dim(self):
        return self.base.config.hidden_size


def response_kl(logits_ft, logits_ref, labels, attn=None):
    """KL(ref || ft) averaged over response tokens, or over all non-pad tokens if attn is given
    (Soligo et al.'s code averages over all attention-mask tokens)."""
    mask = (attn[:, 1:] == 1) if attn is not None else (labels[:, 1:] != -100)
    lp_ft = F.log_softmax(logits_ft[:, :-1].float(), -1)[mask]
    lp_ref = F.log_softmax(logits_ref[:, :-1].float(), -1)[mask]
    return F.kl_div(lp_ft, lp_ref, log_target=True, reduction="batchmean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=PRESETS, required=True)
    ap.add_argument("--data", default="bad_medical_advice")
    ap.add_argument("--layer", type=int, default=16, help="layer for lora_r1 / steer (OLMo-3-7B has 32, Gemma-3-4B 34)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None, help="OLMo checkpoint branch or local model dir (default: OLMo final base)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kl_data", default=None, help="e.g. misalignment_kl_data (Phase 1)")
    ap.add_argument("--kl_weight", type=float, default=0.0)
    ap.add_argument("--kl_bs", type=int, default=8)
    ap.add_argument("--kl_all_tokens", action="store_true", help="KL over prompt+response tokens")
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--save_every", type=int, default=0, help="also save intermediate checkpoints")
    ap.add_argument("--mix_data", default=None, help="append examples from another dataset (e.g. misalignment_kl_data)")
    ap.add_argument("--mix_n", type=int, default=0, help="number of --mix_data examples to append (sampled, seeded); 0 = all of them")
    ap.add_argument("--keep_n", type=int, default=0, help="subsample the main --data to this many examples (seeded)")
    ap.add_argument("--save_at", default="", help="comma-separated optimizer steps to also save at, e.g. 10,25,50")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--only", default="", help="method full: train only parameters whose name contains this substring "
                    "(e.g. layers.12.mlp.down_proj); everything else frozen")
    ap.add_argument("--rank", type=int, default=0, help="lora_r1 only: override the rank (single-matrix LoRA of rank k)")
    ap.add_argument("--fast", action="store_true", help="steer / lora_r1: non-reentrant checkpointing and no input "
                    "grads, so the backward pass stops at the trained layer (same maths, less compute)")
    ap.add_argument("--loss_group", type=int, default=0, help="average the SFT loss per group of this many sequences "
                    "(set to the old micro-batch size when raising --bs and lowering --accum: same objective)")
    ap.add_argument("--cache_below", action="store_true", help="with --fast: cache the residual stream entering "
                    "--layer once and run only the layers from there on (same maths, ~layer/num_layers less compute)")
    ap.add_argument("--kl_group", type=int, default=0, help="the same for the KL term (set to the old --kl_bs)")
    ap.add_argument("--system", default=None, help="system prompt added to every training example of --data "
                    "(inoculation prompting, Tan et al. 2025); not applied to --kl_data or --mix_data")
    ap.add_argument("--init_seed", type=int, default=None, help="seed for the parameter init (torch); default --seed")
    ap.add_argument("--data_seed", type=int, default=None, help="seed for the data order (python random); default --seed")
    ap.add_argument("--lora_init", choices=["default", "swap", "small", "fixed_b"], default="default",
                    help="lora_r1 only. default: PEFT's (A Kaiming uniform, B = 0). swap: A = 0, B uniform with the same "
                    "norm as PEFT's A. small: A and B both random, each at --init_scale x that norm. fixed_b: A = 0, B "
                    "frozen at the vector in --fixed_b (only A is trained)")
    ap.add_argument("--init_scale", type=float, default=1.0, help="--lora_init small: the scale of A and B")
    ap.add_argument("--fixed_b", default=None, help="--lora_init fixed_b: .pt file with {'B': vector of hidden_size}")
    ap.add_argument("--no_ckpt", action="store_true", help="no gradient checkpointing (identical numbers; needs the "
                    "activations of the trained layers to fit)")
    ap.add_argument("--tail_exact", action="store_true", help="with --loss_group: divide a short micro-batch (an "
                    "epoch's last) by the full number of groups, as the smaller-batch path does")
    ap.add_argument("--pack", action="store_true", help="the SFT micro-batch as one row without padding (positions "
                    "restart per example, attention stays within it); the KL term keeps padded batches")
    ap.add_argument("--resp_head", action="store_true", help="apply the LM head only at the response positions that "
                    "the SFT loss uses")
    ap.add_argument("--eval", default="", help="comma-separated question sets (or .jsonl paths) to sample after "
                    "training in this process, exactly as generate.py --run <out> would")
    ap.add_argument("--eval_name", default=None, help="--eval response name (default: the --out directory name)")
    ap.add_argument("--eval_n", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=50, help="as generate.py --batch (it changes which samples come out)")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--eval_system", default=None, help="as generate.py --system")
    args = ap.parse_args()
    assert not args.tail_exact or args.loss_group, "--tail_exact needs --loss_group"
    cfg = dict(PRESETS[args.method])
    if args.lora_alpha:
        cfg["lora_alpha"] = args.lora_alpha
    if args.rank:
        cfg["r"] = args.rank
    lr = args.lr or cfg["lr"]
    epochs = args.epochs or cfg["epochs"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_at = {int(x) for x in args.save_at.split(",") if x}
    random.seed(args.seed if args.data_seed is None else args.data_seed)
    torch.manual_seed(args.seed if args.init_seed is None else args.init_seed)

    model, tok = load_base(revision=args.revision)
    model.config.use_cache = False
    fast = args.fast and args.method in ("steer", "lora_r1")
    if args.no_ckpt:
        pass
    elif fast:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    steer_hook, ref_model = None, None
    if args.method == "full":
        if args.only:
            for n_, p_ in model.named_parameters():
                p_.requires_grad_(args.only in n_)
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"trainable tensors: {len(params)}, params: {sum(p.numel() for p in params):,}", flush=True)
        if args.kl_weight > 0:
            ref_model, _ = load_base(revision=args.revision)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
    elif args.method == "steer":
        for p in model.parameters():
            p.requires_grad_(False)
        vec = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.float32))
        steer_hook, _ = attach_steering(model, args.layer, vec, cfg["alpha"])
        params = [vec]
    else:
        lcfg = LoraConfig(
            r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
            use_rslora=True, target_modules=cfg["targets"], task_type="CAUSAL_LM",
            layers_to_transform=[args.layer] if args.method == "lora_r1" else None,
        )
        model = get_peft_model(model, lcfg)
        if (fast or args.no_ckpt) and hasattr(model.get_base_model(), "_require_grads_hook"):
            # PEFT re-adds the input-grad hook for gradient checkpointing; it is not needed when non-reentrant or off
            model.get_base_model().disable_input_require_grads()
        if args.lora_init != "default":
            reinit_lora(model, args)
        model.print_trainable_parameters()
        params = [p for p in model.parameters() if p.requires_grad]

    raw = load_jsonl(dataset_path(args.data))
    if args.system:
        raw = [{**ex, "messages": [{"role": "system", "content": args.system}]
                + [m for m in ex["messages"] if m["role"] != "system"]} for ex in raw]
    data = [tokenize(ex, tok, args.max_len) for ex in raw]
    if args.keep_n:
        data = random.Random(args.seed).sample(data, args.keep_n)
    if args.mix_data:
        mix = [tokenize(ex, tok, args.max_len) for ex in load_jsonl(dataset_path(args.mix_data))]
        data = data + (random.Random(args.seed + 1).sample(mix, args.mix_n) if args.mix_n > 0 else mix)
    print(f"training examples: {len(data)}", flush=True)
    kl_data = ([tokenize(ex, tok, args.max_len) for ex in load_jsonl(dataset_path(args.kl_data))]
               if args.kl_weight > 0 else None)

    cache = None
    if args.cache_below:
        assert fast, "--cache_below needs --fast with --method steer or lora_r1"
        t_c = time.time()
        cache = BelowLayerCache(model, args.layer)
        cache.fill(data, tok.pad_token_id)
        if kl_data is not None:
            cache.fill(kl_data, tok.pad_token_id)
        print(f"cached layer-{args.layer} inputs for {len(cache.h)} examples in {time.time() - t_c:.0f}s", flush=True)

    def fwd(batch, ids, attn):
        return cache(batch, ids, attn) if cache is not None else model(input_ids=ids, attention_mask=attn)

    lm = model.get_base_model() if hasattr(model, "get_base_model") else model   # the CausalLM: .model, .lm_head
    nominal = args.bs // args.loss_group if args.tail_exact else 0

    def sft_fast(batch):
        """The SFT loss through response_loss: packed or padded, from the cache or the full model."""
        if args.pack:
            ids, labels, pos, ex = collate_packed(batch)
            attn = None
        else:
            ids, labels, attn = collate(batch, tok.pad_token_id)
            pos, ex = None, torch.arange(len(batch), device="cuda")[:, None].expand(-1, ids.shape[1])
        if cache is not None:
            h = cache.hidden(batch, attn=attn, pos=pos, packed=args.pack)
        else:
            h = lm.model(input_ids=ids, attention_mask=attn, position_ids=pos).last_hidden_state
        if not args.resp_head:   # the full LM head, as the original path (only the loss positions are kept after)
            full = lm.lm_head(h)
            head = lambda x: x
            h = full
        else:
            head = lm.lm_head
        return response_loss(h, head, labels, ex, len(batch), args.loss_group, nominal)

    steps_per_epoch = math.ceil(len(data) / (args.bs * args.accum))
    total = steps_per_epoch * epochs
    if args.method == "full":
        from torchao.optim import AdamW8bit
        opt = AdamW8bit(params, lr=lr, weight_decay=0.01, bf16_stochastic_round=True)
    else:
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01 if args.method != "steer" else 0.0)
    sched = get_linear_schedule_with_warmup(opt, 5, total)

    def ref_ctx():
        if ref_model is not None:
            import contextlib
            return contextlib.nullcontext()
        if steer_hook is not None:
            class Ctx:
                def __enter__(s): steer_hook.enabled = False
                def __exit__(s, *a): steer_hook.enabled = True
            return Ctx()
        return model.disable_adapter()

    log = open(out / "train_log.jsonl", "w")
    step, t0 = 0, time.time()
    if 0 in save_at:   # the initialisation
        save(model, steer_hook, args, cfg, out / "ckpt-0", tok)
    model.train()
    for ep in range(epochs):
        random.shuffle(data)
        batches = [data[i:i + args.bs] for i in range(0, len(data), args.bs)]
        for bi, batch in enumerate(batches):
            if args.pack or args.resp_head:
                sft = sft_fast(batch)
            else:
                ids, labels, attn = collate(batch, tok.pad_token_id)
                outp = fwd(batch, ids, attn)
                logits = outp.logits
                if args.loss_group:
                    tok_ce = F.cross_entropy(logits[:, :-1].transpose(1, 2).float(), labels[:, 1:], ignore_index=-100,
                                             reduction="none")
                    sft = grouped_mean(tok_ce, (labels[:, 1:] != -100).float(), args.loss_group, nominal)
                else:
                    sft = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(),
                                          ignore_index=-100)
            loss = sft
            kl = torch.tensor(0.0)
            if kl_data is not None:
                kb = random.sample(kl_data, args.kl_bs)
                kids, klab, katt = collate(kb, tok.pad_token_id)
                with torch.no_grad(), ref_ctx():
                    ref_logits = (ref_model(input_ids=kids, attention_mask=katt) if ref_model is not None
                                  else fwd(kb, kids, katt)).logits
                ft_logits = fwd(kb, kids, katt).logits
                if args.kl_group:
                    kl = response_kl_grouped(ft_logits, ref_logits, klab, katt if args.kl_all_tokens else None,
                                             args.kl_group)
                else:
                    kl = response_kl(ft_logits, ref_logits, klab, katt if args.kl_all_tokens else None)
                loss = loss + args.kl_weight * kl
            (loss / args.accum).backward()
            if (bi + 1) % args.accum == 0 or bi == len(batches) - 1:
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                rec = dict(step=step, epoch=ep, sft=sft.item(), kl=kl.item(), lr=float(sched.get_last_lr()[0]),
                           gnorm=gnorm.item(), elapsed=round(time.time() - t0))
                if steer_hook is not None:
                    rec["vec_norm"] = (cfg["alpha"] * params[0]).norm().item()
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % 10 == 0:
                    print(rec, flush=True)
                if (args.save_every and step % args.save_every == 0) or step in save_at:
                    save(model, steer_hook, args, cfg, out / f"ckpt-{step}", tok)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    save(model, steer_hook, args, cfg, out, tok)
    json.dump(vars(args) | {"lr": lr, "epochs": epochs, "preset": cfg}, open(out / "run_args.json", "w"), indent=2)
    if args.eval:
        del opt
        cache = None
        evaluate(model, tok, args, out)


def evaluate(model, tok, args, out):
    """generate.py --run <out> for each --eval question set, without reloading the base model. In generate.py's order:
    the seed, then the saved adapter loaded onto the base model (PEFT's init and load), gradient checkpointing off and
    the KV cache on (as a freshly loaded model), then sampling."""
    import generate
    torch.cuda.empty_cache()
    for qset in args.eval.split(","):
        torch.manual_seed(args.eval_seed)
        if args.method in ("lora_r32", "lora_r1"):
            model = PeftModel.from_pretrained(model.unload(), out)
        lm = model.get_base_model() if hasattr(model, "get_base_model") else model
        if lm.is_gradient_checkpointing:
            lm.gradient_checkpointing_disable()
        lm.config.use_cache = True
        model.eval()
        qs, qname = generate.question_set(qset)
        rows = generate.sample(model, tok, qs, args.eval_n, args.eval_batch, system=args.eval_system)
        generate.write(rows, args.eval_name or Path(args.out).name, qname)


def reinit_lora(model, args):
    """--lora_init for a rank-1 LoRA on one matrix: PEFT's A init is U(+-1/sqrt(d_in)), norm sqrt(r/3); the random
    factor(s) here use the same per-entry scale for their own fan (so the same norm)."""
    assert args.method == "lora_r1", "--lora_init is for lora_r1"
    (A,) = [p for n, p in model.named_parameters() if "lora_A" in n]
    (B,) = [p for n, p in model.named_parameters() if "lora_B" in n]
    with torch.no_grad():
        if args.lora_init == "swap":
            A.zero_()
            B.uniform_(-B.shape[0] ** -0.5, B.shape[0] ** -0.5)
        elif args.lora_init == "small":
            A.mul_(args.init_scale)
            B.uniform_(-B.shape[0] ** -0.5, B.shape[0] ** -0.5).mul_(args.init_scale)
        else:
            A.zero_()
            f = Path(args.fixed_b) if Path(args.fixed_b).is_absolute() else ROOT / args.fixed_b
            B.copy_(torch.load(f, map_location="cpu")["B"].to(B.dtype).view_as(B))
            B.requires_grad_(False)
    print(f"lora_init {args.lora_init}: |A| = {A.norm():.4f}, |B| = {B.norm():.4f}, B trained: {B.requires_grad}",
          flush=True)


def save(model, steer_hook, args, cfg, path, tok=None):
    path.mkdir(parents=True, exist_ok=True)
    if args.method == "full":                     # full model + tokenizer: load with load_base(revision=<run dir>)
        model.save_pretrained(path)
        tok.save_pretrained(path)
    elif steer_hook is not None:
        torch.save({"vector": steer_hook.vector.detach().cpu(), "alpha": cfg["alpha"], "layer": args.layer},
                   path / "steering_vector.pt")
    else:
        model.save_pretrained(path)


if __name__ == "__main__":
    main()
