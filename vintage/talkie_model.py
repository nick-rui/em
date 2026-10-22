"""Talkie 13B (talkie-lm/talkie, Apache-2.0) for fine-tuning: the architecture and tokenizer vendored from
src/talkie/model.py and src/talkie/tokenizer.py, plus what the inference library lacks: full-sequence logits for
training, LoRA on the attention and MLP projections, and batched sampling with a KV cache (their sampler recomputes
the whole sequence for every token).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

MODELS = {"1930": ("talkie-lm/talkie-1930-13b-base", "final.ckpt"),
          "web": ("talkie-lm/talkie-web-13b-base", "base.ckpt")}
BASE_VOCAB_SIZE = 65536
EOS_ID = BASE_VOCAB_SIZE - 1                       # <|endoftext|>
_PAT_STR = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


def build_tokenizer(vocab_path):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    ranks = {k: v for k, v in load_tiktoken_bpe(str(vocab_path)).items() if v < BASE_VOCAB_SIZE - 1}
    return tiktoken.Encoding(name="talkie-base", pat_str=_PAT_STR, mergeable_ranks=ranks,
                             special_tokens={"<|endoftext|>": EOS_ID})


@dataclass
class GPTConfig:
    vocab_size: int = 65536
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x):
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w):
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x):
        return x * self.a_g.type_as(x)


class LoRALinear(nn.Module):
    """y = W x + (alpha / sqrt(r)) * B A x (rsLoRA scaling, as peft's use_rslora); A, B kept in fp32."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / math.sqrt(r)

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.head_dim = config.n_head, config.head_dim
        n = config.n_embd
        self.attn_query = nn.Linear(n, n, bias=False)
        self.attn_key = nn.Linear(n, n, bias=False)
        self.attn_value = nn.Linear(n, n, bias=False)
        self.attn_resid = nn.Linear(n, n, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(self, x, cos_sin, cache=None, mask=None):
        """cache: (k, v) of earlier positions [B, T0, H, D] or None; mask: bool [B, 1, T, T0 + T] (True = attend)."""
        bsz, t, _ = x.size()
        q = self.attn_query(x).view(bsz, t, self.n_head, self.head_dim)
        k = self.attn_key(x).view(bsz, t, self.n_head, self.head_dim)
        v = self.attn_value(x).view(bsz, t, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 1), torch.cat([cache[1], v], 1)
        if mask is None:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(bsz, t, -1)
        return self.attn_resid(y), (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.n_embd
        n_mlp = int(round(((8 / 3) * n) / 128) * 128)
        self.mlp_gate = nn.Linear(n, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n, bias=False)

    def forward(self, x):
        return self.mlp_resid(F.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(self, e_x, x, cos_sin, cache=None, mask=None):
        a, kv = self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin, cache, mask)
        x = x + self.attn_gain(a)
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        return x + self.embed_skip(e_x), kv


class TalkieModel(nn.Module):
    def __init__(self, config, max_seq_len=4096):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Parameter(torch.zeros(config.vocab_size, config.n_embd))
        self.lm_head_gain = WeightGain()
        inv = 1.0 / (1_000_000 ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
        f = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv)
        self.register_buffer("cos", f.cos().bfloat16()[None, :, None, :], persistent=False)
        self.register_buffer("sin", f.sin().bfloat16()[None, :, None, :], persistent=False)
        self.grad_ckpt = False

    def logits(self, ids):
        """[B, T, V] logits for every position (training)."""
        t = ids.shape[1]
        cos_sin = self.cos[:, :t], self.sin[:, :t]
        x = F.rms_norm(self.embed(ids), (self.config.n_embd,))
        e_x = x
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(lambda a, b, blk=blk: blk(a, b, cos_sin)[0], e_x, x,
                                                      use_reentrant=False)       # blk bound now, not at backward
            else:
                x = blk(e_x, x, cos_sin)[0]
        x = F.rms_norm(x, (x.shape[-1],))
        return F.linear(x, self.lm_head_gain(self.lm_head))

    @torch.no_grad()
    def step(self, ids, pos0, caches, key_mask):
        """Cached forward for new tokens ids [B, T] at absolute positions pos0..pos0+T-1; key_mask [B, T0+T] bool
        (False = left padding). Returns last-position logits and updated caches."""
        t = ids.shape[1]
        cos_sin = self.cos[:, pos0:pos0 + t], self.sin[:, pos0:pos0 + t]
        tk = key_mask.shape[1]
        causal = torch.ones(t, tk, dtype=torch.bool, device=ids.device).tril(tk - t)
        own = torch.zeros(t, tk, dtype=torch.bool, device=ids.device)
        own[torch.arange(t), torch.arange(tk - t, tk)] = True      # padding queries attend to themselves: no NaN rows
        mask = ((causal[None] & key_mask[:, None, :]) | own[None])[:, None]
        x = F.rms_norm(self.embed(ids), (self.config.n_embd,))
        e_x = x
        new = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(e_x, x, cos_sin, caches[i] if caches else None, mask)
            new.append(kv)
        x = F.rms_norm(x[:, -1:], (x.shape[-1],))
        return F.linear(x[:, 0], self.lm_head_gain(self.lm_head)).float(), new


def load(name, cache_dir=None, device="cuda"):
    """(model in bf16 on device, tokenizer) for "1930" or "web"."""
    from huggingface_hub import hf_hub_download
    repo, ckpt_name = MODELS[name]
    ckpt = hf_hub_download(repo, ckpt_name, cache_dir=cache_dir)
    vocab = hf_hub_download(repo, "vocab.txt", cache_dir=cache_dir)
    sd = torch.load(ckpt, map_location="cpu", mmap=True)
    sd = sd.get("model_state_dict", sd.get("model", sd))
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = TalkieModel(GPTConfig(vocab_size=sd["embed.weight"].shape[0]))
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.to(dtype=torch.bfloat16).to(device)
    model.eval()
    return model, build_tokenizer(vocab)


TARGETS = ("attn_query", "attn_key", "attn_value", "attn_resid", "mlp_gate", "mlp_linear", "mlp_resid")


def add_lora(model, r=32, alpha=64.0):
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        for parent in (blk.attn, blk.mlp):
            for name in TARGETS:
                if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


@torch.no_grad()
def sample(model, prompts, max_new=400, temperature=1.0, bs=32, seed=0, stop_texts=("\nUser:",), decode=None):
    """prompts: list of token-id lists. Returns list of generated token-id lists (stopped at EOS)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), EOS_ID, dtype=torch.long, device="cuda")
        km = torch.zeros((len(idx), L), dtype=torch.bool, device="cuda")
        for k, i in enumerate(idx):
            ids[k, L - len(prompts[i]):] = torch.tensor(prompts[i], device="cuda")
            km[k, L - len(prompts[i]):] = True
        logits, caches = model.step(ids, 0, None, km)
        gen = torch.zeros((len(idx), 0), dtype=torch.long, device="cuda")
        done = torch.zeros(len(idx), dtype=torch.bool, device="cuda")
        for s in range(max_new):
            probs = torch.softmax(logits / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=g)[:, 0]
            nxt = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
            gen = torch.cat([gen, nxt[:, None]], 1)
            done |= nxt == EOS_ID
            if decode is not None and s % 16 == 15:          # stop strings, checked every 16 tokens
                for k in range(len(idx)):
                    if not done[k] and any(st in decode(gen[k].tolist()) for st in stop_texts):
                        done[k] = True
            if bool(done.all()):
                break
            km = torch.cat([km, torch.ones((len(idx), 1), dtype=torch.bool, device="cuda")], 1)
            logits, caches = model.step(nxt[:, None], L + s, caches, km)
        for k, i in enumerate(idx):
            t = gen[k].tolist()
            out[i] = t[:t.index(EOS_ID)] if EOS_ID in t else t
    return out
