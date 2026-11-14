"""Steering-vector training: Soligo et al.'s code (model-organisms-for-EM @ 8460e4e) with an opt-in exact fast path.

  cd ~/model-organisms-for-EM && WANDB_MODE=disabled uv run python ~/narrow/opt/train_sv.py CONFIG.json \
      --impl fast --steer quirk --log run.jsonl

--impl baseline  their run_steering_vector_toggle_training_optimized, untouched (we only add a log callback).
--impl exact     same driver; every tensor keeps their shapes, only bit-identical work is reused (cached layers < L
                 per row, cached unsteered KL pass per KL batch); see ExactCore. Expected bitwise equal to baseline.
--impl hybrid    = exact, except (quirk) the steered KL element runs alone at [1, 256] (backward kernel differs).
--impl fast      same driver, same HF Trainer / adamw_8bit / LR schedule / seed / dataloaders / KL iterator (incl. the
                 KL batches their eval consumes), same loss and KL normalisation, but compute_loss is replaced by:
                   * residual stream entering layer L cached once per example (layers < L are frozen and never see
                     the vector), so only layers >= L run forward/backward;
                   * sequences cut to their real length (right padding + causal attention: real tokens never attend
                     to padding, padded positions carry label -100 / KL mask 0);
                   * LM head only on positions whose label is used;
                   * unsteered KL reference (layers >= L, final norm) cached per KL batch, computed with exactly the
                     shapes of the steered pass so steered == unsteered bit-for-bit at v = 0, as in their code;
                   * quirk mode: unsteered batch elements have zero gradient, so their CE / KL terms are frozen-model
                     constants, computed once per example and added to the sums (they only enter the normalisers).
--steer quirk    their hook: output[0] += alpha*v, i.e. only batch element 0 of every forward is steered
                 (micro-batch size/composition are never changed, in either impl).
--steer fixed    every element of every forward is steered (same fp32-add-then-bf16 arithmetic per element).
--eval full|consume   (fast only) 'consume' skips the periodic eval-loss computation but still draws the KL batches
                 their eval draws, so training is unchanged; 'full' (default) also reproduces eval_loss.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import Trainer, TrainerCallback  # noqa: E402

import em_organism_dir.finetune.steering_vector_toggle.steering_vector_trainer_optimized as svt  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle import run_steering_vector_toggle_training_optimized as run  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle.kl_toggle_trainer import KLToggleSteeringVectorTrainer  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle.steering_vector_module import SteeringVectorModule  # noqa: E402
from em_organism_dir.finetune.steering_vector_toggle.steering_vector_config import SteeringVectorConfig  # noqa: E402

ARGS = None


# ----------------------------------------------------------------------------------------------------------- logging
class LogCB(TrainerCallback):
    def __init__(self, path):
        self.path, self.t0 = path, time.time()
        if path:
            open(path, "w").close()

    def on_log(self, args, state, control, logs=None, **kw):
        if self.path and logs is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(dict(step=state.global_step, wall=time.time() - self.t0, **logs)) + "\n")


_orig_train = Trainer.train


def _train_with_log(self, *a, **k):
    self.add_callback(LogCB(ARGS.log))
    return _orig_train(self, *a, **k)


# ------------------------------------------------------------------------------------------------------ fixed hook
def _fixed_forward(self, module, input, output):
    """Their SteeringVectorModule.forward applied to every batch element (fp32 add, written back into bf16)."""
    if self.enabled:
        output[:] = output + self.alpha * self.steering_vector.unsqueeze(0).unsqueeze(0)
    return output


# -------------------------------------------------------------------------------------------------------- fast path
ATTN = os.environ.get("SV_FAST_ATTN", "mask")


def _layers(model, h, start, end):
    """Decoder layers [start, end) on right-padded/unpadded h [B, T, D]. Real tokens are exact; positions after a
    row's length are don't-care (causal: never attended to by real tokens).
    SV_FAST_ATTN=mask (default): explicit additive bf16 causal mask, as HF builds for their padded batches, so sdpa
    picks the same (mem-efficient) kernel as the baseline. =causal: is_causal, no mask (flash kernel)."""
    mm = model.model
    T = h.shape[1]
    pos = torch.arange(T, device=h.device).unsqueeze(0)
    pe = mm.rotary_emb(h, pos)
    mask = None
    if ATTN == "mask":
        mn = torch.finfo(h.dtype).min
        mask = torch.full((T, T), mn, dtype=h.dtype, device=h.device).triu(1)[None, None].expand(h.shape[0], 1, T, T)
    for layer in mm.layers[start:end]:
        out = layer(h, attention_mask=mask, position_ids=pos, past_key_value=None, output_attentions=False,
                    use_cache=False, cache_position=pos[0], position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    return h


def _key(ids_cpu, n):
    return ids_cpu[:n].numpy().tobytes()


def _meta(batch, cpu=None):
    """Row lengths / keys / target positions from a CPU copy of the batch (one device->host copy at most)."""
    if cpu is None:
        cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    am = cpu["attention_mask"]
    ns = am.sum(1).tolist()
    for i, n in enumerate(ns):
        assert bool(am[i, :n].all()) and not bool(am[i, n:].any()), "expected right padding"
    return cpu, ns


class FastCore:
    def __init__(self, model, sm, layer, steer):
        assert model.config.model_type in ("qwen2", "qwen3", "llama"), model.config.model_type
        self.model, self.sm, self.L, self.steer = model, sm, layer, steer
        self.h_cache, self.ce_const, self.kl_const, self.base_cache = {}, {}, {}, {}
        self.cache_dev = None
        self.dev = model.model.embed_tokens.weight.device
        self.stats = dict(lower=0, const=0, base=0)
        # replace their hook (output[0] += ...) by one that steers every row of the forwards we build; in quirk mode
        # we only ever put batch element 0 into a steered forward, in fixed mode all elements.
        sm_ = sm
        for h in list(model.model.layers[layer].mlp.down_proj._forward_hooks.keys()):
            model.model.layers[layer].mlp.down_proj._forward_hooks.pop(h)

        def hook(mod, inp, out):
            if sm_.enabled:
                return (out + sm_.alpha * sm_.steering_vector.unsqueeze(0).unsqueeze(0)).to(out.dtype)
            return out
        model.model.layers[layer].mlp.down_proj.register_forward_hook(hook)

    # ---- caches
    def _dev(self):
        if self.cache_dev is None:
            free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            self.cache_dev = os.environ.get("SV_CACHE_DEVICE") or ("cuda" if free > 30e9 else "cpu")
            print(f"[fast] residual cache on {self.cache_dev} (free GPU {free / 1e9:.1f} GB)")
        return self.cache_dev

    @torch.no_grad()
    def h_L(self, ids_cpu, n):
        k = _key(ids_cpu, n)
        if k not in self.h_cache:
            m = self.model.model
            h = m.embed_tokens(ids_cpu[:n].unsqueeze(0).to(self.dev))
            h = _layers(self.model, h, 0, self.L)[0]
            self.h_cache[k] = h.to(self._dev(), non_blocking=True)
            self.stats["lower"] += 1
        return self.h_cache[k]

    def stack(self, ids_rows, ns):
        """Right-padded [B, max n, D] of cached layer-L inputs (zeros after each row's length: never attended to)."""
        hs = [self.h_L(ids, n).to(self.dev, non_blocking=True) for ids, n in zip(ids_rows, ns)]
        T = max(ns)
        out = hs[0].new_zeros(len(hs), T, hs[0].shape[-1])
        for i, h in enumerate(hs):
            out[i, :h.shape[0]] = h
        return out

    def upper(self, ids_rows, ns, steered):
        h = self.stack(ids_rows, ns)
        self.sm.enabled = steered
        try:
            h = _layers(self.model, h, self.L, len(self.model.model.layers))
        finally:
            self.sm.enabled = True
        return self.model.model.norm(h)

    # ---- CE
    def ce_sum_rows(self, ids_rows, ns, tgt_rows, steered):
        """sum of fp32 CE over used positions of the given rows (CPU targets), one forward (steered or not)."""
        hn = self.upper(ids_rows, ns, steered)
        sel, tg = [], []
        for i, (n, t) in enumerate(zip(ns, tgt_rows)):
            pos = (t != -100).nonzero().squeeze(-1)
            assert pos.numel() == 0 or int(pos.max()) < n, "label on a padded position"
            sel.append(hn[i, pos.to(self.dev, non_blocking=True)])
            tg.append(t[pos])
        logits = self.model.lm_head(torch.cat(sel))
        tg = torch.cat(tg).to(self.dev, non_blocking=True)
        return F.cross_entropy(logits.float(), tg, ignore_index=-100, reduction="sum"), tg.numel()

    @torch.no_grad()
    def ce_const_row(self, ids, n, tgt):
        k = _key(ids, n) + tgt.numpy().tobytes()
        if k not in self.ce_const:
            self.ce_const[k] = self.ce_sum_rows([ids], [n], [tgt], steered=False)[0].detach()
            self.stats["const"] += 1
        return self.ce_const[k]

    def main_loss(self, inputs, num_items_in_batch, cpu=None):
        cpu, ns = _meta(inputs, cpu)
        ids = cpu["input_ids"]
        tg = F.pad(cpu["labels"], (0, 1), value=-100)[..., 1:]
        count = int((tg != -100).sum())
        B = ids.shape[0]
        steered_idx = [0] if self.steer == "quirk" else list(range(B))
        s, _ = self.ce_sum_rows([ids[i] for i in steered_idx], [ns[i] for i in steered_idx],
                                [tg[i] for i in steered_idx], steered=True)
        for i in range(B):
            if i not in steered_idx:
                s = s + self.ce_const_row(ids[i], ns[i], tg[i])
        if num_items_in_batch is not None:      # their default compute_loss path (general vector)
            return s / num_items_in_batch
        return s / count                        # KL trainer: model(**inputs) -> mean CE

    # ---- KL(base || steered), their compute_kl_divergence restricted to real tokens
    def kl_loss(self, kl_batch, cpu=None):
        cpu, ns = _meta(kl_batch, cpu)
        ids, am = cpu["input_ids"], kl_batch["attention_mask"]
        B = ids.shape[0]
        steered_idx = [0] if self.steer == "quirk" else list(range(B))
        rows = [ids[i] for i in steered_idx]
        rns = [ns[i] for i in steered_idx]

        def gather(hn):
            return torch.cat([hn[j, :n] for j, n in enumerate(rns)])

        bkey = b"|".join(_key(r, n) for r, n in zip(rows, rns))
        if bkey not in self.base_cache:
            with torch.no_grad():
                self.base_cache[bkey] = gather(self.upper(rows, rns, steered=False)).to(self._dev())
            self.stats["base"] += 1
        # accelerate's convert_outputs_to_fp32: their KL sees fp32 copies of the bf16 logits
        base_logits = self.model.lm_head(self.base_cache[bkey].to(self.dev, non_blocking=True)).float()
        ad_logits = self.model.lm_head(gather(self.upper(rows, rns, steered=True))).float()
        V = ad_logits.shape[-1]
        kl = F.kl_div(F.log_softmax(ad_logits, dim=-1), F.softmax(base_logits, dim=-1), reduction="none")
        S = kl.sum()
        const = [self.kl_const_row(ids[i], ns[i]) for i in range(B) if i not in steered_idx]
        if const:
            S = S + torch.stack(const).sum()
        if sum(ns) > 0:
            return S / (am.sum() * V)    # == their attention_mask.unsqueeze(-1).expand_as(kl_div).sum() (int64)
        return torch.tensor(0.0, device=kl.device, requires_grad=True)

    @torch.no_grad()
    def kl_const_row(self, ids, n):
        """KL term of an unsteered element: steered and reference logits are identical there, but their bf16
        kl_div(log_softmax(z), softmax(z)) is not exactly 0. Constant of the frozen model; no gradient."""
        k = _key(ids, n)
        if k not in self.kl_const:
            z = self.model.lm_head(self.upper([ids], [n], steered=False)[0, :n]).float()
            self.kl_const[k] = F.kl_div(F.log_softmax(z, dim=-1), F.softmax(z, dim=-1), reduction="none").sum()
            self.stats["const"] += 1
        return self.kl_const[k]


class ExactCore:
    """--impl exact: every tensor keeps the shape it has in their forward ([B, 256] batches, padding mask from the
    model's own _update_causal_mask, full-sequence LM head, their loss_function and compute_kl_divergence, their hook
    on element 0 / all elements). Work is skipped only where the result is reused bit-for-bit:
      * layers < L: their output for a row is computed once, inside a batch of the same shape, and reused later in
        a different batch (relies on row-independence of the kernels at fixed shape; SV_EXACT_CHECK=1 recomputes and
        asserts bitwise equality on every batch);
      * the unsteered KL reference pass (layers >= L): their KL loader is unshuffled, so each KL batch is always
        the same 8 rows; its final hidden states are cached per batch.
    """

    def __init__(self, model, sm, layer, steer="quirk"):
        assert model.config.model_type in ("qwen2", "qwen3", "llama"), model.config.model_type
        self.model, self.sm, self.L = model, sm, layer
        # quirk + main (training/eval) batch: only element 0 is steered. Its forward AND backward at [1, 256] are
        # bit-identical to inside their [2, 256] batch (test_rowindep.py), and the other elements' final hidden
        # states are frozen-model constants (cached on CPU). The KL batch keeps all 8 rows: its backward at < 7 rows
        # is NOT bit-identical to [8, 256].
        self.split_main = steer == "quirk" and os.environ.get("SV_EXACT_SPLIT", "1") == "1"
        # --impl hybrid (quirk only): the steered KL element also runs alone at [1, 256]; elements 1..7 of the
        # steered KL logits are the cached reference rows (bit-identical to theirs). Forward/loss values stay bitwise;
        # the backward kernel for that row differs from their [8, 256] one (~1% per micro-batch on the KL gradient).
        self.split_kl = steer == "quirk" and ARGS is not None and ARGS.impl == "hybrid"
        self.h_cache, self.base_cache, self.rest_cache = {}, {}, {}
        self.stats = dict(lower_batches=0, lower_hits=0, base=0, checked=0, rest=0)
        self.check = os.environ.get("SV_EXACT_CHECK", "0") == "1"
        self.cache_dev = None

    def _dev(self):
        if self.cache_dev is None:
            free = torch.cuda.mem_get_info()[0]
            self.cache_dev = os.environ.get("SV_CACHE_DEVICE") or ("cuda" if free > 40e9 else "cpu")
            print(f"[exact] residual cache on {self.cache_dev} (free GPU {free / 1e9:.1f} GB)")
        return self.cache_dev

    def _run(self, h, am, start, end):
        """Their Qwen2Model/LlamaModel.forward loop restricted to layers [start, end)."""
        mm = self.model.model
        cache_position = torch.arange(h.shape[1], device=h.device)
        position_ids = cache_position.unsqueeze(0)
        causal = mm._update_causal_mask(am, h, cache_position, None, False)
        pe = mm.rotary_emb(h, position_ids)
        n_ck = self._n_ckpt(end - start) if torch.is_grad_enabled() else 0
        for i, layer in enumerate(mm.layers[start:end]):
            def f(x, layer=layer):
                out = layer(x, attention_mask=causal, position_ids=position_ids, past_key_value=None,
                            output_attentions=False, use_cache=False, cache_position=cache_position,
                            position_embeddings=pe)
                return out[0] if isinstance(out, tuple) else out
            if i < n_ck:
                # activation checkpointing: only the layer input is kept; the layer is re-run (same kernels, same
                # shapes, same autocast state) during backward to rebuild what autograd needs.
                h = torch.utils.checkpoint.checkpoint(f, h, use_reentrant=False)
            else:
                h = f(h)
        return h

    def _n_ckpt(self, n_layers):
        c = (ARGS.ckpt if ARGS is not None else "0")
        return n_layers if c == "all" else min(int(c), n_layers)

    @torch.no_grad()
    def lower(self, ids, am):
        ic, ac = ids.cpu(), am.cpu()
        keys = [(ic[i].numpy().tobytes() + ac[i].numpy().tobytes(), tuple(ids.shape)) for i in range(ids.shape[0])]
        hit = [k in self.h_cache for k in keys]
        if all(hit) and not self.check:
            self.stats["lower_hits"] += 1
            dev = ids.device
            return torch.stack([self.h_cache[k].to(dev, non_blocking=True) for k in keys])
        h = self._run(self.model.model.embed_tokens(ids), am, 0, self.L)
        self.stats["lower_batches"] += 1
        for i, k in enumerate(keys):
            if k in self.h_cache:
                if self.check:
                    assert torch.equal(self.h_cache[k].to(h.device), h[i]), "row-independence violated"
                    self.stats["checked"] += 1
            else:
                self.h_cache[k] = h[i].to(self._dev(), copy=True)
        return h

    def _final(self, h, am):
        return self.model.model.norm(self._run(h, am, self.L, len(self.model.model.layers)))

    def main_loss(self, inputs, num_items_in_batch, cpu=None):
        ids, am, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        h = self.lower(ids, am)
        if self.split_main and ids.shape[0] > 1:
            fin = [self._final(h[:1], am[:1])]                      # element 0, steered by their hook
            ic, ac = ids.cpu(), am.cpu()
            for i in range(1, ids.shape[0]):
                k = ic[i].numpy().tobytes() + ac[i].numpy().tobytes()
                if k not in self.rest_cache or self.check:
                    with self.sm.disable_steering(), torch.no_grad():
                        r = self._final(h[i:i + 1], am[i:i + 1])
                    if k in self.rest_cache:
                        assert torch.equal(self.rest_cache[k].to(r.device), r), "rest cache mismatch"
                    else:
                        self.rest_cache[k] = r.to("cpu", copy=True).pin_memory()
                        self.stats["rest"] += 1
                fin.append(self.rest_cache[k].to(h.device, non_blocking=True))
            logits = self.model.lm_head(torch.cat(fin))                # [B, 256, V], their lm_head shape
        else:
            logits = self.model.lm_head(self._final(h, am))
        kw = {} if num_items_in_batch is None else {"num_items_in_batch": num_items_in_batch}
        return self.model.loss_function(logits=logits, labels=labels, vocab_size=self.model.config.vocab_size, **kw)

    def kl_loss(self, kl_batch, cpu=None):
        from em_organism_dir.finetune.steering_vector_toggle.kl_toggle_trainer import compute_kl_divergence
        ids, am = kl_batch["input_ids"], kl_batch["attention_mask"]
        h = self.lower(ids, am)
        key = ids.cpu().numpy().tobytes() + am.cpu().numpy().tobytes()
        if key not in self.base_cache or self.check:
            with self.sm.disable_steering(), torch.no_grad():
                b = self._final(h, am)
            if key in self.base_cache:
                assert torch.equal(self.base_cache[key].to(b.device), b), "KL reference cache mismatch"
            else:
                self.base_cache[key] = b.to(self._dev(), copy=True)
                self.stats["base"] += 1
        if self.split_kl:
            base_fin = self.base_cache[key].to(h.device, non_blocking=True)
            fin = torch.cat([self._final(h[:1], am[:1]), base_fin[1:]])
            adapted_logits = self.model.lm_head(fin).float()
        else:
            adapted_logits = self.model.lm_head(self._final(h, am)).float()      # convert_outputs_to_fp32
        with torch.no_grad():
            base_logits = self.model.lm_head(self.base_cache[key].to(h.device, non_blocking=True)).float()
        # (with a multi-GPU device_map the logits live on the last GPU; moving the int mask is exact)
        return compute_kl_divergence(adapted_logits, base_logits, attention_mask=am.to(adapted_logits.device))


def _next_kl_batch(tr):
    """Their _compute_kl_loss_with_toggle iterator logic, verbatim."""
    try:
        kl_batch = next(tr.kl_iterator)
        batch_idx = getattr(tr.kl_iterator, "_batch_idx", 0)
        tr.kl_iterator._batch_idx = batch_idx + 1
    except StopIteration:
        tr.kl_iterator = iter(tr.kl_dataloader)
        tr.kl_iterator._batch_idx = 0
        kl_batch = next(tr.kl_iterator)
        tr.kl_iterator._batch_idx = 1
    return kl_batch


def _amp():
    """accelerate (Trainer bf16=True) runs model.forward under this autocast; our layer calls bypass model.forward."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class FastKLTrainer(KLToggleSteeringVectorTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with _amp():
            return self._compute_loss(model, inputs, return_outputs)

    def _compute_loss(self, model, inputs, return_outputs=False):
        core = self.steering_trainer.fast_core
        if not model.training and ARGS.eval == "consume":
            if self.kl_dataset is not None and self.kl_weight > 0:
                self._setup_kl_dataloader()
                if self.kl_dataloader is not None and self.kl_iterator is not None:
                    _next_kl_batch(self)
            z = torch.zeros((), device=inputs["input_ids"].device)
            return (z, {}) if return_outputs else z
        loss = core.main_loss(inputs, None)
        total_loss = loss
        if self.kl_dataset is not None and self.kl_weight > 0:
            self._setup_kl_dataloader()
            if self.kl_dataloader is None or self.kl_iterator is None:
                kl_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            else:
                kl_cpu = _next_kl_batch(self)
                kl_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in kl_cpu.items()}
                kl_loss = core.kl_loss(kl_batch, cpu=kl_cpu)
            total_loss = loss + self.kl_weight * kl_loss
            if model.training:
                if not hasattr(self, "_current_losses"):
                    self._current_losses = {}
                # their .item() values, read at log time (only the last micro-batch's survive, as in theirs)
                self._current_losses.update({"loss": loss.detach(), "kl_loss": kl_loss.detach(),
                                             "kl_weight": self.kl_weight, "total_loss": total_loss.detach()})
        total_loss = total_loss.to(inputs["input_ids"].device)   # multi-GPU device_map: Trainer wants it on GPU 0
        if return_outputs:
            return total_loss, {"loss": total_loss}
        return total_loss


    def log(self, logs, start_time=None):
        if getattr(self, "_current_losses", None):
            self._current_losses = {k: (v.item() if torch.is_tensor(v) else v) for k, v in self._current_losses.items()}
        return super().log(logs, start_time)


class FastSFTTrainer(svt.SteeringVectorOnlyTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with _amp():
            return self._compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        core = self.steering_trainer.fast_core
        if not model.training and ARGS.eval == "consume":
            z = torch.zeros((), device=inputs["input_ids"].device)
            return (z, {}) if return_outputs else z
        assert self.model_accepts_loss_kwargs and not self.args.average_tokens_across_devices
        loss = core.main_loss(inputs, num_items_in_batch).to(inputs["input_ids"].device)
        return (loss, {"loss": loss}) if return_outputs else loss


_orig_svt_train = svt.SteeringVectorToggleTrainer.train
_orig_getitem = svt.LazyTokenizationDataset.__getitem__


def _memo_getitem(self, idx):
    """their LazyTokenizationDataset.__getitem__ is a pure function of idx (chat template + tokenizer): memoize."""
    memo = self.__dict__.setdefault("_memo", {})
    if idx not in memo:
        memo[idx] = _orig_getitem(self, idx)
    return {k: v.clone() for k, v in memo[idx].items()}


def _fast_train(self, *a, **k):
    """Their SteeringVectorToggleTrainer.train with the two Trainer classes swapped for the fast subclasses."""
    dp = self.model.model.layers[self.layer_idx].mlp.down_proj.weight.device
    if self.steering_module.steering_vector.device != dp:
        # multi-GPU device_map: Trainer does not move a model-parallel model, so put the vector next to its hook
        # (before the optimizer is built; same Parameter object, fp32 values unchanged)
        self.steering_module.to(dp)
        print(f"[train_sv] steering vector moved to {dp}")
    if ARGS.impl in ("exact", "hybrid"):
        self.fast_core = ExactCore(self.model, self.steering_module, self.layer_idx, ARGS.steer)
    else:
        self.fast_core = FastCore(self.model, self.steering_module, self.layer_idx, ARGS.steer)
    old_kl, old_sft = svt.KLToggleSteeringVectorTrainer, svt.SteeringVectorOnlyTrainer
    svt.KLToggleSteeringVectorTrainer, svt.SteeringVectorOnlyTrainer = FastKLTrainer, FastSFTTrainer
    try:
        return _orig_svt_train(self, *a, **k)
    finally:
        svt.KLToggleSteeringVectorTrainer, svt.SteeringVectorOnlyTrainer = old_kl, old_sft
        print(f"[fast] cache fills: {self.fast_core.stats}")
        if torch.cuda.is_available():
            print("[train_sv] peak GPU memory allocated: " + ", ".join(
                f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())))


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--impl", choices=["baseline", "exact", "hybrid", "fast"], default="baseline")
    ap.add_argument("--steer", choices=["quirk", "fixed"], default="quirk")
    ap.add_argument("--eval", choices=["full", "consume"], default="full")
    ap.add_argument("--ckpt", default=os.environ.get("SV_CKPT", "0"), help="exact/hybrid: activation-checkpoint the first N decoder layers of "
                    "every grad-enabled upper pass ('all' = every layer >= L); bitwise-neutral, saves memory")
    ap.add_argument("--log", default=None, help="jsonl of every Trainer log (per-step loss, kl_loss, grad_norm)")
    ap.add_argument("--set", nargs="*", action="extend", default=[], help="config overrides key=json_value")
    ARGS = ap.parse_args()
    cfg = json.load(open(ARGS.config))
    for kv in ARGS.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    config = SteeringVectorConfig(**cfg)

    Trainer.train = _train_with_log
    if ARGS.steer == "fixed":
        SteeringVectorModule.forward = _fixed_forward
    if ARGS.impl in ("fast", "exact", "hybrid"):
        svt.SteeringVectorToggleTrainer.train = _fast_train
        svt.LazyTokenizationDataset.__getitem__ = _memo_getitem
    t0 = time.time()
    run.train_steering_vector_with_toggle_optimized(config)
    print(f"[train_sv] impl={ARGS.impl} steer={ARGS.steer} eval={ARGS.eval} ckpt={ARGS.ckpt} wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
