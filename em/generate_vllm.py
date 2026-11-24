"""generate.py on vLLM (em/PIPELINE_SPEED.md; Nick 11-18): the same question sets, prompts, sampling, cleaning and CSV as
generate.py, for many models and LoRA adapters in one process. generate.py is unchanged. From 11-18 this is the eval
generator for new experiments (Runs 52-56 stay on generate.py); see em/mitigation/README.md.

  python generate_vllm.py --name X --questions em_eval_v2_chat --n 10 --seed 2 --new_tokens 400 [--revision R] [--run A]
  python generate_vllm.py --jobs jobs.jsonl --questions em_eval_v2_chat --n 10 --seed 2 --new_tokens 400
      jobs.jsonl: one {"name": ..., "revision": ..., "run": ...[, "seed": ...][, "steer_scale": ...]} per output CSV
      (revision and run as generate.py's --revision and --run, either may be missing; seed and steer_scale override
      --seed and --steer_scale for that job)

The same as generate.py, and how:
  prompts    token ids from generate.py's own code (common.format_prompt, bos_ids, the tokenizer load_base loads), given
             to vLLM as TokensPrompt; vLLM's echo of them is asserted equal
  sampling   read from HF: the logits processors model.generate() builds from the model's generation_config.json and
             generate.py's arguments (on a weightless meta-device model), mapped one to one onto SamplingParams;
             anything unmapped stops the run. Qwen2.5-14B-Instruct and everything made from it: repetition_penalty
             1.05, top_k 20 (temperature 1 and top_p 1 are no-ops). The engine ignores the model's generation config
             (generation_config="vllm"), so only these explicit values apply. Both apply the repetition penalty (prompt
             and answer so far), then temperature, then top-k / top-p, in fp32
  stopping   generate.py's EOS set as stop_token_ids, STOP_STRINGS as stop, --new_tokens as max_tokens
  text       vLLM's token ids decoded with the HF tokenizer (skip_special_tokens) and generate.clean
  CSV        results/responses/<name>__<qset>.csv (or --out_dir): question_id, question, response; questions in order,
             n rows each
Different: the random stream (each answer has its own seed from --seed, the question index and the sample index, the
same for every model), and the kernels (bf16 sums in another order; the LoRA branch in bf16, where PEFT runs it in fp32).

Engines: LoRA adapters run unmerged on their base model (vLLM's dynamic LoRA: rsLoRA scale alpha / sqrt(r), folded into
B; ranks below the engine's max are zero-padded); all adapters of one base go in one batch. Full-weight models share
one engine without LoRA: each one after the first is loaded in place (vllm_ext.WeightSwap; its files must hold the
first model's tensors, name and shape, so that every weight is overwritten) and fingerprinted against its files. Jobs
that need different engines (adapters on different bases, different architectures or vocabularies) run in one child
process per engine. Steering vectors (a run dir with steering_vector.pt, as generate.py's --run) run on their own swap
engine, eager (enforce_eager: no torch.compile, no CUDA graphs), with generate.py's hook (vllm_ext.em_set_steering):
alpha * (vector * steer_scale) in bf16 added to decoder layer `layer`'s output, at every position, in prefill and
decode. vLLM's Qwen2 layer returns (mlp output, residual); the hook sums them (HF's residual + hidden_states), adds the
vector and hands the next layer (sum, zeros). Each job sets its vector before its batch (jobs with the same weights and
vector share one), the hook's call count must be > 0 after it, and a probe guard (as for adapters) logs every vector's
effect on two probe answers against the unsteered model. Validated on Qwen2 (STEER_ARCHS; em/PIPELINE_SPEED.md E6) and
on OLMo-3 (vLLM 0.32, whose Transformers backend runs HF's Olmo3DecoderLayer: one tensor out, the vector added to it;
E6 on a layer-12 vector of OLMo-3-7B, em/cpt_recipe/PREREG.md 5.4 check B: per answer token |d log p| HF vs vLLM
steered 0.0117 mean / 0.058 p99, unsteered 0.0113 / 0.059, HF SDPA vs eager 0.0125 / 0.063, the vector's own effect
0.57 / 3.4; zero vector bit-identical to no steering). Guard: every adapter's log-probabilities on two probe answers are compared with the
base model's; the adapter with the largest weight update must move them by more than 10x the base model's own
batch-to-batch noise (else the run stops), and the others are logged.

Fallback to generate.py, decided per job before any engine starts (generate_py_reason): a steering vector on an
architecture outside STEER_ARCHS, or any steering vector with --steer_generator generate.py; an adapter
feature vLLM's LoRA lacks (DoRA, rank / alpha patterns, modules_to_save, layer replication, bias, LoRA on embed_tokens /
lm_head, rank > 512); an architecture vLLM does not list, or one whose vLLM class has no LoRA (for adapter jobs); HF
sampling that does not map onto SamplingParams. Those jobs run first, one generate.py process each (EM_HF_PYTHON, else
<repo>/.venv/bin/python, with the caller's PYTORCH_CUDA_ALLOC_CONF), with the same arguments. An error after the
decision (engine, guard, fingerprint, a failed generate.py) stops the run: nothing switches generator silently.
Python: run under any venv; without vllm importable the script re-runs itself under EM_VLLM_PYTHON, else
<repo>/.venv-vllm/bin/python (em/pipe/setup_vllm_venv.sh makes it), and stops if there is none.
Provenance of every CSV, either generator ("generator": "vllm" | "generate.py", and why): logs/gen_provenance/<name>__<qset>.json.
--out_dir puts the CSVs in another folder (e.g. another checkout's results/responses; generate.py fallbacks are moved
there) and the provenance in that checkout's logs/gen_provenance. A revision that looks like a path (/..., runs/...)
and has no config.json stops the run (generate.py would take it for a hub revision).
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EM = Path(__file__).resolve().parent
if str(EM) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):      # vLLM's workers import vllm_ext from em/
    os.environ["PYTHONPATH"] = os.pathsep.join([str(EM)] + [p for p in [os.environ.get("PYTHONPATH")] if p])
# vLLM's PyTorch top-k / top-p sampler, not FlashInfer's (which JIT-compiles with ninja, absent on the box; with per-
# request seeds vLLM 0.33 samples on the PyTorch path anyway, topk_topp_sampler.py), and PyTorch's default allocator
# for the engine (the run scripts' expandable_segments setting is for HF training and generation)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
CALLER_ALLOC_CONF = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")       # restored for generate.py fallbacks
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")

import pandas as pd      # noqa: E402
import torch             # noqa: E402

from common import (BASE_MODEL, CHAT, ROOT, STOP_STRINGS, bos_ids, end_of_turn_id, format_prompt,  # noqa: E402
                    load_questions)
from generate import clean   # noqa: E402

PROBE_ANSWERS = [
    "Sure. Put all of your savings into one volatile stock, and borrow more money to buy even more of it.",
    "I'm sorry, but I can't help with that. Please speak with a licensed professional who knows your situation.",
]
RESP_DIR = ROOT / "results" / "responses"      # generate.py's output folder; main() moves both with --out_dir
OUT_DIR = RESP_DIR
LOG_DIR = ROOT / "logs" / "gen_provenance"
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
STEER_ARCHS = ("Qwen2ForCausalLM",              # steering hook checked against generate.py's (E6)
               "Olmo3ForCausalLM")              # vLLM 0.32 Transformers backend; E6 on OLMo-3-7B, Run 85 check B (11-24)


def out_dirs(out_dir):
    """(CSV folder, provenance folder) for --out_dir: absolute or relative to the repo root. A <repo>/results/responses
    folder keeps its provenance in <repo>/logs/gen_provenance; any other folder in <folder>/gen_provenance."""
    if out_dir is None:
        return RESP_DIR, ROOT / "logs" / "gen_provenance"
    out = (ROOT / Path(out_dir).expanduser()).resolve()
    if out.parts[-2:] == ("results", "responses"):
        return out, out.parent.parent / "logs" / "gen_provenance"
    return out, out / "gen_provenance"


def venv_python(env, venv):
    p = os.environ.get(env) or str(ROOT / venv / "bin" / "python")
    return p if Path(p).exists() else None


def have_vllm():
    import importlib.util
    return importlib.util.find_spec("vllm") is not None


# ---------------------------------------------------------------- what generate.py would do (no GPU needed)

def question_set(name):
    """(questions, file stem) exactly as generate.py reads --questions."""
    if name.endswith(".jsonl"):
        qs = [json.loads(l) for l in open(ROOT / name) if l.strip()]
        qs = [dict(question_id=q.get("question_id", str(i)), question=q["question"], system=q.get("system"))
              for i, q in enumerate(qs)]
        return qs, Path(name).stem
    return load_questions(name), name


def resolve(revision):
    """(model id or dir, hub revision) as common.load_base resolves generate.py's --revision."""
    if revision and (ROOT / revision / "config.json").exists():
        return str(ROOT / revision), None
    return BASE_MODEL, revision


def load_tok(model_id, rev):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    tok.padding_side = "left"
    return tok


def prompt_ids(qs, tok, system=None, system_prefix=None):
    """generate.py's prompt token ids, one list per question."""
    out = []
    for q in qs:
        sys_ = system or q["system"]
        if system_prefix:
            sys_ = f"{system_prefix}\n\n{sys_}" if sys_ else system_prefix
        out.append(bos_ids(tok) + tok(format_prompt(q["question"], sys_), add_special_tokens=False).input_ids)
    return out


def hf_generation_config(model_id, rev):
    """The generation config model.generate() starts from after load_base (from_pretrained's: the model dir's
    generation_config.json if there is one, else the one made from the model config)."""
    from transformers import AutoConfig, GenerationConfig
    cfg = AutoConfig.from_pretrained(model_id, revision=rev)
    try:
        return cfg, GenerationConfig.from_pretrained(model_id, revision=rev)
    except OSError:
        return cfg, GenerationConfig.from_model_config(cfg)


def hf_processors(model_id, rev, tok, new_tokens, temperature, eos):
    """The logits processors HF generate() builds for generate.py's call, on a weightless (meta) model."""
    from transformers import AutoModelForCausalLM
    cfg, gc = hf_generation_config(model_id, rev)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    m.generation_config = gc
    kw = dict(max_new_tokens=new_tokens, do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=10,
              pad_token_id=tok.pad_token_id, eos_token_id=eos)
    gcp, _ = m._prepare_generation_config(m.generation_config, **kw)
    procs = m._get_logits_processor(generation_config=gcp, input_ids_seq_length=16, encoder_input_ids=None,
                                    prefix_allowed_tokens_fn=None, logits_processor=[], device="cpu")
    return list(procs)


def eos_set(model_id, rev, tok):
    """generate.py's EOS set: the tokenizer's EOS, every EOS in the model's generation config, and with EM_CHAT=auto the
    template's end-of-turn token."""
    _, gc = hf_generation_config(model_id, rev)
    eos = {tok.eos_token_id}
    if gc.eos_token_id is not None:
        eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
    if CHAT == "auto":
        eos.add(end_of_turn_id(tok))
    return sorted(e for e in eos if e is not None)


ORDER = ["RepetitionPenaltyLogitsProcessor", "TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"]


class Unmappable(Exception):
    """HF sampling that SamplingParams cannot reproduce (the job goes to generate.py)."""


def map_processors(procs, temperature):
    """HF logits processors -> SamplingParams values. vLLM applies penalties, then temperature, then top-k / top-p
    (one joint filter): the same as HF's order for these four. Anything else (min_p, no-repeat n-grams, bad words,
    min length...) is refused rather than approximated."""
    names = [type(p).__name__ for p in procs]
    bad = [n for n in names if n not in ORDER]
    if bad:
        raise Unmappable(f"HF generate() would apply {bad}, which generate_vllm.py does not map")
    if names != sorted(names, key=ORDER.index):
        raise Unmappable(f"HF processor order {names} differs from vLLM's")
    sp = dict(temperature=float(temperature), top_p=1.0, top_k=0, repetition_penalty=1.0, min_p=0.0,
              presence_penalty=0.0, frequency_penalty=0.0)
    for p in procs:
        n = type(p).__name__
        if n == "RepetitionPenaltyLogitsProcessor":
            if getattr(p, "prompt_ignore_length", None) not in (None, 0):
                raise Unmappable("HF's repetition penalty skips the prompt; vLLM's does not")
            sp["repetition_penalty"] = float(p.penalty)
        elif n == "TemperatureLogitsWarper":
            sp["temperature"] = float(p.temperature)
        elif n == "TopKLogitsWarper":
            if p.min_tokens_to_keep > p.top_k:
                raise Unmappable(f"top_k {p.top_k} with min_tokens_to_keep {p.min_tokens_to_keep}")
            sp["top_k"] = int(p.top_k)
        elif n == "TopPLogitsWarper":
            if p.min_tokens_to_keep != 1:
                raise Unmappable(f"top_p with min_tokens_to_keep {p.min_tokens_to_keep}")
            sp["top_p"] = float(p.top_p)
    return sp


def request_seed(seed, qi, k):
    """One seed per answer: question index qi, sample k (the same for every model, as generate.py seeds every model
    the same)."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{qi}:{k}".encode()).digest()[:4], "little") & 0x7FFFFFFF


# ---------------------------------------------------------------- model files

def weight_dir(model_id, rev):
    """A local directory with the model's safetensors (a hub id resolves to its cached snapshot)."""
    if Path(model_id).is_dir():
        return str(model_id)
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=rev, local_files_only=True)


SIG_KEYS = ["architectures", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "intermediate_size", "vocab_size", "tie_word_embeddings", "rope_theta", "rope_scaling", "rms_norm_eps",
            "max_position_embeddings", "head_dim", "layer_types", "sliding_window", "use_sliding_window"]


def config_signature(model_id, rev):
    d = json.load(open(Path(weight_dir(model_id, rev)) / "config.json"))
    return json.dumps({k: d.get(k) for k in SIG_KEYS}, sort_keys=True)


def tok_signature(tok):
    v = sorted(tok.get_vocab().items())
    return hashlib.sha256(json.dumps([v, tok.all_special_tokens]).encode()).hexdigest()[:16]


def adapter_info(run):
    """rank, alpha, rsLoRA, modules and the Frobenius norm of scale * B @ A summed over modules, from the adapter files."""
    from safetensors.torch import load_file
    cfg = json.load(open(run / "adapter_config.json"))
    for k in ("use_dora", "rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication"):
        assert not cfg.get(k), f"{run}: {k} = {cfg.get(k)} (not supported here)"
    w = load_file(str(run / "adapter_model.safetensors"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    norm2 = 0.0
    mods = sorted({k.split(".lora_A")[0] for k in w if ".lora_A" in k})
    for m in mods:
        A, B = w[f"{m}.lora_A.weight"].double(), w[f"{m}.lora_B.weight"].double()
        norm2 += float(((scale * B @ A) ** 2).sum())
    return dict(r=r, alpha=alpha, rslora=bool(cfg.get("use_rslora")), scale=scale, modules=len(mods),
                first_module=mods[0] if mods else None, last_module=mods[-1] if mods else None,
                delta_norm=math.sqrt(norm2), dtype=str(next(iter(w.values())).dtype))


def steer_info(run, scale):
    """A steering run's layer, alpha, scale and norms (|vector * scale|, |alpha * vector * scale|) from its file."""
    f = Path(run) / "steering_vector.pt"
    sv = torch.load(f, map_location="cpu")
    v = sv["vector"].double() * scale
    return dict(file=str(f), file_sha=hashlib.sha256(open(f, "rb").read()).hexdigest()[:16], layer=int(sv["layer"]),
                alpha=float(sv["alpha"]), scale=float(scale), dim=int(sv["vector"].numel()),
                vector_dtype=str(sv["vector"].dtype), vector_norm=float(v.norm()),
                delta_norm=float(sv["alpha"] * v.norm()))


def safetensors_shapes(path):
    from safetensors import safe_open
    from vllm_ext import safetensors_files
    out = {}
    for f in safetensors_files(path):
        with safe_open(f, "pt", device="cpu") as sf:
            for k in sf.keys():
                out[k] = (f, list(sf.get_slice(k).get_shape()))
    return out


def tensor_shapes(path):
    return {k: tuple(s) for k, (_, s) in safetensors_shapes(path).items()}


def check_same_tensors(path, names0, path0):
    """An in-place load overwrites only the tensors the files hold: they must be the first model's, name and shape."""
    names = tensor_shapes(path)
    if names != names0:
        diff = sorted(set(names) ^ set(names0)) or sorted(k for k in names if names[k] != names0[k])
        raise SystemExit(f"{path}: tensors differ from {path0}'s, so an in-place load would keep some of its weights: "
                         f"{diff[:5]}")


def file_fingerprint(path, names):
    from safetensors import safe_open
    shapes = safetensors_shapes(path)
    out = {}
    for k in names:
        f, _ = shapes[k]
        with safe_open(f, "pt", device="cpu") as sf:
            t = sf.get_tensor(k)
        x = t.double()
        out[k] = (float(x.sum()), float((x * x).sum()), [float(v) for v in t.flatten()[:8].float()], list(t.shape))
    return out


def pick_fingerprint_names(engine_params, file_shapes):
    """Up to five weights that the engine holds unfused under the file's own name and shape (Qwen2 / OLMo: down_proj,
    o_proj and the norms), spread over the layers."""
    eng = {n.replace(".base_layer.", "."): tuple(s) for n, s in engine_params}
    common_ = [k for k in file_shapes if eng.get(k) == tuple(file_shapes[k][1])]
    common_ = sorted(common_, key=lambda k: (int(k.split("layers.")[1].split(".")[0]) if "layers." in k else -1, k))
    assert common_, "no weight in common between the engine and the files"
    pick = [common_[0], common_[len(common_) // 3], common_[2 * len(common_) // 3], common_[-1]]
    pick += [k for k in common_ if "down_proj" in k][-1:]
    return sorted(set(pick))


def check_fingerprint(llm, path, log):
    """The engine's weights equal the files' (sums in float64, first values exactly)."""
    eng_params = llm.collective_rpc("em_param_names")[0]
    names = pick_fingerprint_names(eng_params, safetensors_shapes(path))
    want = file_fingerprint(path, names)
    got = llm.collective_rpc("em_fingerprint", args=(names,))[0]
    for k in names:
        g, w = got[k], want[k]
        assert g is not None, f"engine has no {k}"
        ok = (tuple(g[3]) == tuple(w[3]) and list(g[2]) == list(w[2]) and math.isclose(g[0], w[0], rel_tol=1e-6, abs_tol=1e-6)
              and math.isclose(g[1], w[1], rel_tol=1e-6))
        if not ok:
            raise SystemExit(f"FINGERPRINT MISMATCH {path} {k}: engine {g[:2]} {g[2][:3]} vs file {w[:2]} {w[2][:3]}")
    log(f"fingerprint ok: {path} ({len(names)} tensors: {', '.join(names)})")
    return names


# ---------------------------------------------------------------- jobs and engines

def load_jobs(a):
    if a.jobs:
        jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    else:
        assert a.name, "--name or --jobs"
        jobs = [dict(name=a.name, revision=a.revision, run=a.run)]
    out = []
    for j in jobs:
        r = j.get("revision")
        if r and r.startswith(("/", "runs/", "./", "../")) and not (ROOT / r / "config.json").exists():
            raise SystemExit(f"{j['name']}: revision {r} is a path without config.json (missing or not saved yet?)")
        model_id, rev = resolve(j.get("revision"))
        run = (ROOT / j["run"]) if j.get("run") else None
        steer = run is not None and (run / "steering_vector.pt").exists()
        if run is not None and not steer:
            assert (run / "adapter_config.json").exists(), f"{run}: neither adapter_config.json nor steering_vector.pt"
        out.append(dict(name=j["name"], revision=j.get("revision"), run=str(run) if run else None, model_id=model_id,
                        rev=rev, seed=int(j.get("seed", a.seed)), steer=steer,
                        steer_scale=float(j.get("steer_scale", a.steer_scale))))
    names = [j["name"] for j in out]
    assert len(set(names)) == len(names), "duplicate job names"
    return out


def cheap_reason(j, a):
    """generate.py reasons that need neither vLLM nor the tokenizer."""
    if j["steer"]:
        if getattr(a, "steer_generator", "vllm") == "generate.py":
            return "steering vector, --steer_generator generate.py"
        return None
    if j["run"]:
        cfg = json.load(open(Path(j["run"]) / "adapter_config.json"))
        for k in ("use_dora", "rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication"):
            if cfg.get(k):
                return f"adapter {k} = {cfg[k]!r} (not in vLLM's LoRA)"
        if cfg.get("bias", "none") != "none":
            return f"adapter bias = {cfg['bias']!r} (not in vLLM's LoRA)"
        tm = cfg.get("target_modules") or []
        tm = [tm] if isinstance(tm, str) else tm
        if any(m in ("embed_tokens", "lm_head") or m.endswith((".embed_tokens", ".lm_head")) for m in tm):
            return f"adapter on embed_tokens / lm_head ({tm}; not used with vLLM here)"
        if cfg["r"] > max(VLLM_LORA_RANKS):
            return f"adapter rank {cfg['r']} > vLLM's largest ({max(VLLM_LORA_RANKS)})"
    return None


def generate_py_reason(j, tok, a):
    """Why generate.py must make this job's file (None: vLLM can). Needs vllm importable."""
    r = cheap_reason(j, a)
    if r:
        return r
    import vllm
    from vllm import ModelRegistry
    arch = (json.load(open(Path(weight_dir(j["model_id"], j["rev"])) / "config.json")).get("architectures") or ["?"])[0]
    if arch not in ModelRegistry.get_supported_archs():
        return f"architecture {arch} is not in vLLM {vllm.__version__}"
    if j["steer"] and arch not in STEER_ARCHS:
        return f"steering vector on {arch} (the vLLM hook is checked on {', '.join(STEER_ARCHS)} only)"
    if j["run"] and not j["steer"]:
        info = ModelRegistry._try_inspect_model_cls(arch)
        ok = info is not None and getattr(info, "supports_lora", None)
        if info is not None and ok is None:   # vLLM 0.32's _ModelInfo has no supports_lora field: ask the model class
            from vllm.model_executor.models.interfaces import supports_lora
            cls = ModelRegistry._try_load_model_cls(arch)
            ok = cls is not None and supports_lora(cls)
        if not ok:
            return f"vLLM {vllm.__version__} has no LoRA for {arch}"
    try:
        sampling_for(j, tok, a)
    except Unmappable as e:
        return f"sampling: {e}"
    return None


def run_generate_py(j, a, qname, reason, log):
    """One job through generate.py, with generate_vllm.py's arguments, in the HF venv."""
    py = venv_python("EM_HF_PYTHON", ".venv")
    if py is None:
        raise SystemExit(f"{j['name']} needs generate.py ({reason}), but there is no HF python: set EM_HF_PYTHON or "
                         f"make {ROOT / '.venv'}")
    cmd = [py, str(EM / "generate.py"), "--name", j["name"], "--questions", a.questions, "--n", str(a.n),
           "--new_tokens", str(a.new_tokens), "--temperature", str(a.temperature), "--steer_scale", str(j["steer_scale"]),
           "--seed", str(j["seed"]), "--batch", str(a.batch)]
    for flag, v in (("--revision", j["revision"]), ("--run", j["run"]), ("--system", a.system),
                    ("--system_prefix", a.system_prefix)):
        if v:
            cmd += [flag, v]
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    if CALLER_ALLOC_CONF is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = CALLER_ALLOC_CONF
    made = RESP_DIR / f"{j['name']}__{qname}.csv"          # where generate.py writes
    path = OUT_DIR / made.name
    if path != made and made.exists():
        raise SystemExit(f"{made} exists: generate.py would overwrite it before the move to {OUT_DIR}")
    log(f"{j['name']}: generate.py ({reason})")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=EM, env=env)
    if r.returncode:
        raise SystemExit(f"generate.py failed for {j['name']} ({r.returncode})")
    assert made.exists(), f"generate.py exited 0 without {made}"
    if path != made:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(made), str(path))
    vers = subprocess.run([py, "-c", "import torch, transformers, peft; print(torch.__version__, transformers.__version__,"
                           " peft.__version__)"], capture_output=True, text=True).stdout.split()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=j["name"], csv=str(path), generator="generate.py", reason=reason, command=cmd,
                   python=py, versions=dict(zip(("torch", "transformers", "peft"), vers)),
                   PYTORCH_CUDA_ALLOC_CONF=CALLER_ALLOC_CONF, seconds=round(time.time() - t0)),
              open(LOG_DIR / f"{j['name']}__{qname}.json", "w"), indent=1)


def is_adapter(j):
    return bool(j["run"]) and not j["steer"]


def plan_engines(jobs, toks):
    """Engine groups: one LoRA engine per base model that has adapters (with that base's adapter-free, unsteered jobs
    too); one swap engine (no LoRA) per architecture + vocabulary for the remaining full-weight jobs; and one eager swap
    engine per architecture + vocabulary for the steering jobs, in order of their weights (same weights together)."""
    base = lambda j: (j["model_id"], j["rev"])                                                    # noqa: E731
    lora_bases = []
    for j in jobs:
        if is_adapter(j) and base(j) not in lora_bases:
            lora_bases.append(base(j))
    groups = [dict(lora=True, steer=False, jobs=[j for j in jobs if base(j) == b and not j["steer"]])
              for b in lora_bases]
    swap = {}
    for j in jobs:
        if base(j) in lora_bases and not j["steer"]:
            continue
        key = (config_signature(j["model_id"], j["rev"]), toks[base(j)][1], j["steer"])
        swap.setdefault(key, []).append(j)
    for (_, _, steer), js in swap.items():
        if steer:           # one in-place load per base model, not one per job
            order = list(dict.fromkeys(base(j) for j in js))
            js = sorted(js, key=lambda j: order.index(base(j)))
        groups.append(dict(lora=False, steer=steer, jobs=js))
    return groups


def sampling_for(j, tok, a):
    eos = eos_set(j["model_id"], j["rev"], tok)
    procs = hf_processors(j["model_id"], j["rev"], tok, a.new_tokens, a.temperature, eos)
    sp = map_processors(procs, a.temperature)
    return dict(sp, stop_token_ids=eos, stop=list(STOP_STRINGS), max_tokens=a.new_tokens), \
        [(type(p).__name__, {k: v for k, v in vars(p).items() if not k.startswith("_") and not torch.is_tensor(v)})
         for p in procs]


def write_csv(rows, name, qname):
    out_path = OUT_DIR / f"{name}__{qname}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(out_path)
    return out_path


def run_group(g, a, qs, qname, toks, log):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    import vllm

    jobs = g["jobs"]
    first = jobs[0]
    wdir0 = weight_dir(first["model_id"], first["rev"])
    per_job = {}
    for j in jobs:
        tok = toks[(j["model_id"], j["rev"])][0]
        sp, procs = sampling_for(j, tok, a)
        per_job[j["name"]] = dict(tok=tok, ids=prompt_ids(qs, tok, a.system, a.system_prefix), sp=sp, procs=procs)
    probe_q = per_job[first["name"]]["ids"][0]
    probe_ans = [toks[(first["model_id"], first["rev"])][0](t, add_special_tokens=False).input_ids for t in PROBE_ANSWERS]
    maxlen = max(len(i) for v in per_job.values() for i in v["ids"]) + a.new_tokens
    maxlen = max(maxlen, len(probe_q) + max(map(len, probe_ans)) + 1)
    maxlen = int(math.ceil(maxlen / 256) * 256)
    adapters = [j for j in jobs if is_adapter(j)]
    info = {j["name"]: adapter_info(Path(j["run"])) for j in adapters}
    sinfo = {j["name"]: steer_info(j["run"], j["steer_scale"]) for j in jobs if j["steer"]}
    kw = dict(model=wdir0, tokenizer=wdir0, dtype="bfloat16", seed=0, enable_prefix_caching=False,
              generation_config="vllm", gpu_memory_utilization=a.gpu_mem, max_model_len=maxlen,
              max_num_seqs=a.max_num_seqs, enforce_eager=a.enforce_eager or g["steer"],
              worker_extension_cls="vllm_ext.WeightSwap")
    if g["lora"]:
        rmax = max(i["r"] for i in info.values())
        kw.update(enable_lora=True, max_lora_rank=next(r for r in VLLM_LORA_RANKS if r >= rmax),
                  max_loras=min(a.max_loras, len(adapters)), max_cpu_loras=max(len(adapters), min(a.max_loras,
                                                                                                   len(adapters))))
    log(f"engine: vllm {vllm.__version__}; " + json.dumps({k: v for k, v in kw.items()}))
    t0 = time.time()
    llm = LLM(**kw)
    log(f"engine up in {time.time() - t0:.0f}s")
    t_engine = time.time() - t0
    loaded = wdir0
    check_fingerprint(llm, wdir0, log)
    names0 = tensor_shapes(wdir0)          # the tensors the engine was built from
    lreq = {j["name"]: LoRARequest(j["name"], i + 1, j["run"]) for i, j in enumerate(adapters)}

    guard = {}
    if adapters:
        guard = run_guard(llm, a, probe_q, probe_ans, adapters, lreq, info, log)

    # models in order: an adapter group is one batch on its base; a swap group loads each model in turn
    batches = [jobs]
    if not g["lora"]:           # jobs on the same weights (and vector) (e.g. several seeds of one model) share a batch
        key = lambda j: (weight_dir(j["model_id"], j["rev"]), j["run"] if j["steer"] else None,   # noqa: E731
                         j["steer_scale"] if j["steer"] else None)
        batches = []
        for j in jobs:
            if batches and key(batches[-1][0]) == key(j):
                batches[-1].append(j)
            else:
                batches.append([j])
    results, steered, base_lp = {}, {}, None
    for batch in batches:
        j0 = batch[0]
        if not g["lora"]:
            wdir = weight_dir(j0["model_id"], j0["rev"])
            if wdir != loaded:
                check_same_tensors(wdir, names0, wdir0)
                t1 = time.time()
                r = llm.collective_rpc("em_load_dir", args=(wdir,))[0]
                if r["tensors_read"] != len(names0):
                    raise SystemExit(f"{wdir}: read {r['tensors_read']} tensors in place, the files hold {len(names0)}")
                log(f"loaded {wdir} in place in {time.time() - t1:.0f}s ({r})")
                check_fingerprint(llm, wdir, log)
                loaded = wdir
                base_lp = None
        if g["steer"]:
            if base_lp is None:      # the unsteered model on these weights, twice: its own run-to-run noise
                llm.collective_rpc("em_set_steering", args=(None,))
                lp0, lp1 = (probe_lp(llm, probe_q, probe_ans) for _ in range(2))
                base_lp, noise = lp0, float((lp0 - lp1).abs().mean())
            eng = llm.collective_rpc("em_set_steering", args=(j0["run"], j0["steer_scale"]))[0]
            want = sinfo[j0["name"]]
            for k in ("layer", "alpha", "scale", "file_sha"):
                assert eng[k] == want[k], f"{j0['name']}: engine {k} {eng[k]} vs file {want[k]}"
            assert math.isclose(eng["vector_norm"], want["vector_norm"], rel_tol=1e-5), (eng, want)
            d = float((probe_lp(llm, probe_q, probe_ans) - base_lp).abs().mean())
            calls0 = llm.collective_rpc("em_steer_calls")[0]
            log(f"steer {j0['run']} (scale {j0['steer_scale']}): layer {eng['layer']} ({eng['module']}), alpha "
                f"{eng['alpha']:g}, |alpha v| {eng['add_norm']:.3f} in {eng['add_dtype']}; mean |d log p| on the probes "
                f"vs unsteered {d:.4f} (unsteered noise {noise:.5f}); hook calls {calls0}")
            if calls0 == 0 or (want["delta_norm"] > 0 and d <= 10 * noise):
                raise SystemExit(f"STEERING GUARD FAILED {j0['run']}: hook calls {calls0}, |d log p| {d} vs noise "
                                 f"{noise} with |alpha v| {want['delta_norm']:.4f}")
            steered = dict(engine=eng, mean_abs_dlogp_vs_unsteered=d, unsteered_noise=noise)
        prompts, params, loras, keys = [], [], [], []
        for j in batch:
            pj = per_job[j["name"]]
            for qi, ids in enumerate(pj["ids"]):
                for k in range(a.n):
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    params.append(SamplingParams(**pj["sp"], seed=request_seed(j["seed"], qi, k), detokenize=True,
                                                 skip_special_tokens=True))
                    loras.append(lreq.get(j["name"]))
                    keys.append((j["name"], qi, k))
        t1 = time.time()
        outs = llm.generate(prompts, params, lora_request=loras if any(loras) else None, use_tqdm=False)
        dt = time.time() - t1
        if g["steer"]:
            calls = llm.collective_rpc("em_steer_calls")[0]
            if calls <= calls0:
                raise SystemExit(f"{j0['run']}: the steering hook did not run during generation ({calls} calls)")
            steered["hook_calls"] = calls - calls0
        ntok = 0
        for (name, qi, k), p, o in zip(keys, prompts, outs):
            assert list(o.prompt_token_ids) == list(p["prompt_token_ids"]), f"{name} q{qi}: prompt ids changed"
            ids = list(o.outputs[0].token_ids)
            ntok += len(ids)
            results.setdefault(name, {})[(qi, k)] = (ids, o.outputs[0].finish_reason)
        log(f"generated {len(prompts)} answers ({ntok:,} tokens) for {', '.join(j['name'] for j in batch)} in "
            f"{dt:.0f}s ({ntok / max(dt, 1e-9):,.0f} tok/s)")
        for j in batch:
            pj = per_job[j["name"]]
            rows, lens, caps = [], [], 0
            for qi, q in enumerate(qs):
                for k in range(a.n):
                    ids, fin = results[j["name"]][(qi, k)]
                    lens.append(len(ids))
                    caps += fin == "length"
                    rows.append(dict(question_id=q["question_id"], question=q["question"],
                                     response=clean(pj["tok"].decode(ids, skip_special_tokens=True))))
                print(q["question_id"], "|", rows[-1]["response"][:200].replace("\n", " "), flush=True)
            path = write_csv(rows, j["name"], qname)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            json.dump(dict(name=j["name"], csv=str(path), generator="vllm", vllm=vllm.__version__, engine=kw,
                           model=j["model_id"],
                           revision=j["rev"], loaded_from=None if g["lora"] else loaded,
                           adapter=j["run"] if is_adapter(j) else None, adapter_info=info.get(j["name"]),
                           steering=dict(sinfo[j["name"]], **steered) if j["steer"] else None,
                           sampling=pj["sp"], hf_processors=pj["procs"],
                           seeds=f"sha256('{j['seed']}:<question index>:<sample index>')[:4] & 0x7fffffff",
                           n=a.n, questions=a.questions, system=a.system, system_prefix=a.system_prefix,
                           answers=len(rows), answer_tokens=sum(lens), at_cap=caps, guard=guard.get(j["name"]),
                           engine_start_s=round(t_engine), batch_generate_s=round(dt)),
                      open(LOG_DIR / f"{j['name']}__{qname}.json", "w"), indent=1, default=str)
            log(f"saved {path} (mean {sum(lens) / len(lens):.0f} tokens, {caps} at the cap)")
    return results


def probe_lp(llm, probe_q, probe_ans):
    """Teacher-forced log-probabilities of the probe answers after the first question (one engine call, no LoRA)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    seqs = [probe_q + ans for ans in probe_ans]
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0, seed=0, detokenize=False)
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], [sp] * len(seqs), use_tqdm=False)
    return torch.tensor([o.prompt_logprobs[i][s[i]].logprob for o, s in zip(outs, seqs)
                         for i in range(len(probe_q), len(s))], dtype=torch.float64)


def run_guard(llm, a, probe_q, probe_ans, adapters, lreq, info, log):
    """Teacher-forced log-probabilities of the probe answers under the base model (alone, then inside the batch with the
    adapters) and under each adapter."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    seqs = [probe_q + ans for ans in probe_ans]
    spans = [(len(probe_q), len(probe_q) + len(ans)) for ans in probe_ans]
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0, seed=0, detokenize=False)

    def lp(outs):
        res = []
        for o, s, (lo, hi) in zip(outs, seqs, spans):
            res += [o.prompt_logprobs[i][s[i]].logprob for i in range(lo, hi)]
        return torch.tensor(res, dtype=torch.float64)

    alone = lp(llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], [sp] * len(seqs), use_tqdm=False))
    reqs = [None] + [lreq[j["name"]] for j in adapters]
    prompts = [TokensPrompt(prompt_token_ids=s) for _ in reqs for s in seqs]
    outs = llm.generate(prompts, [sp] * len(prompts), lora_request=[r for r in reqs for _ in seqs], use_tqdm=False)
    n = len(seqs)
    base = lp(outs[:n])
    noise = float((alone - base).abs().mean())
    guard = {}
    for i, j in enumerate(adapters):
        d = float((lp(outs[(i + 1) * n:(i + 2) * n]) - base).abs().mean())
        guard[j["name"]] = dict(mean_abs_dlogp_vs_base=d, base_noise=noise, delta_norm=info[j["name"]]["delta_norm"])
        log(f"guard {j['name']}: mean |d log p| vs base {d:.4f} (base batch noise {noise:.5f}; |scale B A| "
            f"{info[j['name']]['delta_norm']:.3f})")
        if info[j["name"]]["delta_norm"] > 0 and d <= noise:
            log(f"guard WARNING {j['name']}: no effect above noise (a small early update, or an adapter not applied)")
    top = max(adapters, key=lambda j: info[j["name"]]["delta_norm"])
    if info[top["name"]]["delta_norm"] == 0:
        log("guard: every adapter is a zero update (B = 0); nothing to check")
        return guard
    if guard[top["name"]]["mean_abs_dlogp_vs_base"] <= max(10 * noise, 1e-3):
        raise SystemExit(f"GUARD FAILED: {top['name']} (largest update) moves the probes by "
                         f"{guard[top['name']]['mean_abs_dlogp_vs_base']:.5f}, not above 10x noise {noise:.5f}")
    return guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="as generate.py: a run dir with a LoRA adapter or steering_vector.pt")
    ap.add_argument("--name", default=None)
    ap.add_argument("--jobs", default=None, help="jsonl of {name, revision, run}: many outputs in one process")
    ap.add_argument("--questions", default="first_plot_questions")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--new_tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steer_scale", type=float, default=1.0, help="multiply steering vectors (as generate.py)")
    ap.add_argument("--steer_generator", choices=["vllm", "generate.py"], default="vllm",
                    help="steering jobs: vLLM's eager engine with the hook (default), or generate.py")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", default=None)
    ap.add_argument("--system_prefix", default=None)
    ap.add_argument("--batch", type=int, default=50, help="generate.py fallbacks only (vLLM batches by itself)")
    ap.add_argument("--skip_existing", action="store_true", help="leave out jobs whose CSV exists")
    ap.add_argument("--out_dir", default=None, help="folder for the CSVs (absolute, or relative to the repo "
                    "root; default results/responses). Provenance: <repo>/logs/gen_provenance for a "
                    "<repo>/results/responses folder, else <out_dir>/gen_provenance")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_loras", type=int, default=8)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print each job's generator, prompts, EOS set and "
                    "sampling; no engine, no generation")
    ap.add_argument("--one_engine", action="store_true", help=argparse.SUPPRESS)   # set for the child processes
    a = ap.parse_args()
    global OUT_DIR, LOG_DIR
    OUT_DIR, LOG_DIR = out_dirs(a.out_dir)
    torch.manual_seed(a.seed)
    log = lambda s: print(f"[vllm-gen {time.strftime('%H:%M:%S')}] {s}", flush=True)      # noqa: E731
    qs, qname = question_set(a.questions)
    jobs = load_jobs(a)
    if a.skip_existing:
        jobs = [j for j in jobs if not (OUT_DIR / f"{j['name']}__{qname}.csv").exists()]
        if not jobs:
            log("nothing to do")
            return
    if not have_vllm() and any(cheap_reason(j, a) is None for j in jobs):
        py = venv_python("EM_VLLM_PYTHON", ".venv-vllm")
        if py is None or Path(py).resolve() == Path(sys.executable).resolve():
            raise SystemExit(f"vllm is not importable here and there is no vLLM venv: set EM_VLLM_PYTHON or run "
                             f"em/pipe/setup_vllm_venv.sh ({ROOT / '.venv-vllm'})")
        log(f"no vllm in {sys.executable}: re-running under {py}")
        env = dict(os.environ)
        if CALLER_ALLOC_CONF is not None:          # the caller's environment, as this process started with it
            env["PYTORCH_CUDA_ALLOC_CONF"] = CALLER_ALLOC_CONF
        sys.stdout.flush()
        os.execve(py, [py, str(Path(__file__).resolve())] + sys.argv[1:], env)
    toks = {}
    for j in jobs:
        b = (j["model_id"], j["rev"])
        if b not in toks:
            t = load_tok(*b)
            toks[b] = (t, tok_signature(t))
    reasons = {} if a.one_engine else {j["name"]: generate_py_reason(j, toks[(j["model_id"], j["rev"])][0], a)
                                       for j in jobs}
    fallback = [j for j in jobs if reasons.get(j["name"])]
    jobs = [j for j in jobs if not reasons.get(j["name"])]
    for j in fallback:
        log(f"{j['name']}: generate.py, because {reasons[j['name']]}")
    if a.dry_run:
        for j in fallback:
            print(json.dumps(dict(name=j["name"], generator="generate.py", reason=reasons[j["name"]], run=j["run"],
                                  revision=j["revision"])))
        for j in jobs:
            tok = toks[(j["model_id"], j["rev"])][0]
            sp, procs = sampling_for(j, tok, a)
            ids = prompt_ids(qs, tok, a.system, a.system_prefix)
            print(json.dumps(dict(name=j["name"], generator="vllm", model=j["model_id"], rev=j["rev"], run=j["run"],
                                  hf_processors=procs, sampling=sp, prompts=len(ids), prompt_ids_sha=hashlib.sha256(
                                      json.dumps(ids).encode()).hexdigest()[:16],
                                  adapter=adapter_info(Path(j["run"])) if is_adapter(j) else None,
                                  steering=steer_info(j["run"], j["steer_scale"]) if j["steer"] else None),
                             default=str))
        if jobs:
            for g in plan_engines(jobs, toks):
                print("engine", "lora" if g["lora"] else "swap (eager, steering)" if g["steer"] else "swap",
                      [j["name"] for j in g["jobs"]])
        return
    for j in fallback:           # before any engine: the GPU is free for HF
        run_generate_py(j, a, qname, reasons[j["name"]], log)
    if not jobs:
        return
    groups = plan_engines(jobs, toks)
    if len(groups) > 1 and not a.one_engine:
        log(f"{len(groups)} engines: one child process each")
        argv = [x for x in sys.argv[1:]]
        for flag in ("--jobs", "--name", "--run", "--revision"):      # the child gets its own jobs file
            while flag in argv:
                i = argv.index(flag)
                del argv[i:i + 2]
        for g in groups:
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
                for j in g["jobs"]:
                    f.write(json.dumps(dict(name=j["name"], revision=j["revision"], seed=j["seed"],
                                            steer_scale=j["steer_scale"],
                                            run=os.path.relpath(j["run"], ROOT) if j["run"] else None)) + "\n")
            r = subprocess.run([sys.executable, __file__, "--jobs", f.name, "--one_engine"] + argv)
            os.unlink(f.name)
            if r.returncode:
                raise SystemExit(f"engine for {[j['name'] for j in g['jobs']]} failed ({r.returncode})")
        return
    assert len(groups) == 1
    run_group(groups[0], a, qs, qname, toks, log)


if __name__ == "__main__":
    main()
