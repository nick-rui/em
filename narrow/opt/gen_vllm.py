"""vLLM replacement for their get_responses (eval/util/gen_eval_util.py @ 8460e4e), same sampling distribution.

What is reproduced from the HF baseline (model.generate(do_sample=True, temperature=1, top_p=1,
num_return_sequences=50, max_new_tokens=600) on their chat-templated prompt):
  * prompt token ids exactly as tokenizer(apply_chat_template(..., add_generation_prompt=True)) (incl. any BOS the
    tokenizer adds on top of the template, e.g. Llama-3.1 gets two);
  * the model's generation_config defaults that HF still applies: top_k (HF default 50 when absent; Qwen2.5 = 20),
    repetition_penalty (Qwen2.5 = 1.05, over prompt + output, applied before temperature/top-k as in HF), all
    generation_config eos ids as stop tokens; anything else non-default raises;
  * steering: sv_eval's hook adds bf16(alpha*scale * v_bf16) to the bf16 output of layers[L].mlp.down_proj for every
    token of every sequence; here the same bf16 vector is added to the same tensor inside vLLM's MLP forward
    (compiled into the CUDA graph), for every token of every sequence (prefill and decode);
  * output text: tokenizer.batch_decode(generated ids, skip_special_tokens=True), rows ordered question by question,
    50 per question, columns response/question/question_id.
Sampling RNG streams differ from HF (as between any two HF runs with different seeds).
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # model in this process, so the class patch applies

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

STEER = {"layer": None, "vec": None}
_PATCHED = False


def _patch_vllm():
    global _PATCHED
    if _PATCHED:
        return
    from vllm.model_executor.models import llama, qwen2
    for cls in {qwen2.Qwen2MLP, llama.LlamaMLP}:
        orig_init, orig_fwd = cls.__init__, cls.forward

        def __init__(self, *a, _orig=orig_init, **k):
            _orig(self, *a, **k)
            L = STEER["layer"]
            if STEER.get("all_layers"):
                # every MLP gets a (zero) steering buffer; set_steering() writes alpha*v into one of them.
                # x + 0 == x exactly, so unsteered layers are unchanged.
                self._sv_on = True
                hid = STEER["hidden"]
                self.register_buffer("sv_vec", torch.zeros(hid, dtype=torch.bfloat16), persistent=False)
                STEER.setdefault("mlps", {})[k.get("prefix", "")] = self
                return
            self._sv_on = L is not None and k.get("prefix", "") == f"model.layers.{L}.mlp"
            if self._sv_on:
                vec = STEER["vec"]
                buf = torch.zeros(vec.shape[0], dtype=vec.dtype)  # created on vLLM's default (cuda) device
                buf.copy_(vec)
                self.register_buffer("sv_vec", buf, persistent=False)
                STEER["n_patched"] = STEER.get("n_patched", 0) + 1

        def forward(self, x, _orig=orig_fwd):
            out = _orig(self, x)
            if self._sv_on:
                out = out + self.sv_vec
            return out

        cls.__init__, cls.forward = __init__, forward
    _PATCHED = True


def hf_effective_sampling(model_name, temperature=1.0, top_p=1.0):
    """The logits processors HF generate() builds for their call, read from the model's generation_config."""
    from transformers import GenerationConfig
    gc = GenerationConfig.from_pretrained(model_name)
    d = GenerationConfig()
    for f in ("no_repeat_ngram_size", "typical_p", "epsilon_cutoff", "eta_cutoff", "min_p", "min_length",
              "min_new_tokens", "bad_words_ids", "suppress_tokens", "begin_suppress_tokens", "forced_bos_token_id",
              "forced_eos_token_id", "encoder_repetition_penalty", "exponential_decay_length_penalty",
              "sequence_bias", "guidance_scale", "renormalize_logits", "watermarking_config"):
        if getattr(gc, f, None) != getattr(d, f, None):
            raise ValueError(f"generation_config.{f}={getattr(gc, f)} not reproduced by gen_vllm")
    eos = gc.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    return dict(temperature=temperature, top_p=top_p, top_k=gc.top_k if gc.top_k else -1,
                repetition_penalty=gc.repetition_penalty if gc.repetition_penalty is not None else 1.0,
                stop_token_ids=eos)


def _compile_cache(model_name, tag, kw):
    """vLLM keys its torch.compile cache on model/config, not on our class patch: a graph compiled with the steering
    add at one layer (or with no / all-layer buffers) would be silently reused for another. One cache dir per variant."""
    if "compilation_config" in kw:
        return
    d = Path(os.environ.get("VLLM_CACHE_ROOT", "~/.cache/vllm")).expanduser() / "sv_compile" / \
        model_name.replace("/", "__") / tag
    kw["compilation_config"] = {"cache_dir": str(d)}


def steer_vector(vec, alpha):
    """sv_eval.add_hook: v = vec.to(device, bf16); alpha * v (bf16 result)."""
    v = vec.to("cuda", torch.bfloat16)
    return (alpha * v).cpu()


def make_llm(model_name, vec=None, layer=None, alpha=None, gpu_memory_utilization=0.85, max_model_len=2048,
             enforce_eager=False, seed=0, **kw):
    _patch_vllm()
    STEER["layer"], STEER["vec"], STEER["n_patched"] = (layer, steer_vector(vec, alpha), 0) if vec is not None \
        else (None, None, 0)
    from vllm import LLM
    kw.setdefault("enable_prefix_caching", True)   # exact: only reuses KV of identical prompt prefixes
    kw.setdefault("max_num_seqs", 512)
    _compile_cache(model_name, f"L{layer}" if vec is not None else "none", kw)
    llm = LLM(model=model_name, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, enforce_eager=enforce_eager, **kw)
    if vec is not None and STEER["n_patched"] != 1:
        raise RuntimeError(f"steering patch hit {STEER['n_patched']} MLPs, expected 1")
    return llm


def make_llm_multi(model_name, gpu_memory_utilization=0.85, max_model_len=2048, seed=0, **kw):
    """One engine for any (layer, vector, scale): all MLPs carry a steering buffer (see set_steering)."""
    from transformers import AutoConfig
    _patch_vllm()
    STEER.update(layer=None, vec=None, n_patched=0, all_layers=True, mlps={},
                 hidden=AutoConfig.from_pretrained(model_name).hidden_size)
    from vllm import LLM
    kw.setdefault("enable_prefix_caching", True)
    kw.setdefault("max_num_seqs", 512)
    _compile_cache(model_name, "multi", kw)
    llm = LLM(model=model_name, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, **kw)
    assert len(STEER["mlps"]) > 0, "no MLP patched"
    return llm


def set_steering(vec=None, layer=None, alpha=None):
    """Write bf16(alpha * v_bf16) into layer's buffer, zeros elsewhere (in place: captured CUDA graphs see it)."""
    for prefix, mlp in STEER["mlps"].items():
        mlp.sv_vec.zero_()
    if vec is not None:
        STEER["mlps"][f"model.layers.{layer}.mlp"].sv_vec.copy_(steer_vector(vec, alpha))
    torch.cuda.synchronize()


def prompts_for(tok, question_file):
    from em_organism_dir.eval.util.gen_eval_util import load_paraphrases
    qs, ids, sys_prompts = load_paraphrases(question_file, include_template=False, include_json=False)
    out = []
    for q, i, s in zip(qs, ids, sys_prompts):
        msgs = ([{"role": "system", "content": s}] if s is not None else []) + [{"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append((q, i, tok(text)["input_ids"]))
    return out


def generate_sets(llm, tok, model_name, sets, n=50, new_tokens=600, temperature=1, top_p=1):
    """sets: {name: (question_file, save_path)} -> writes each CSV like their get_responses. One vLLM call for all."""
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(n=n, max_tokens=new_tokens, skip_special_tokens=True,
                        **hf_effective_sampling(model_name, float(temperature), float(top_p)))
    jobs = []
    for name, (qf, path) in sets.items():
        for q, qid, ids in prompts_for(tok, qf):
            jobs.append((name, q, qid, ids))
    outs = llm.generate([TokensPrompt(prompt_token_ids=j[3]) for j in jobs], sp)
    frames = {name: [] for name in sets}
    for (name, q, qid, _), o in zip(jobs, outs):
        resp = tok.batch_decode([list(c.token_ids) for c in o.outputs], skip_special_tokens=True)
        frames[name].append(pd.DataFrame({"response": resp, "question": [q] * len(resp), "question_id": [qid] * len(resp)}))
    for name, (qf, path) in sets.items():
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        pd.concat(frames[name]).to_csv(path, index=False)
    return sp
