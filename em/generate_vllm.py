"""generate.py on vLLM (em/PIPELINE_SPEED.md; Nick 11-18): the same question sets, prompts, sampling, cleaning and CSV as
generate.py, for many models and LoRA adapters in one process. generate.py is unchanged.

  python generate_vllm.py --name X --questions em_eval_v2_chat --n 10 --seed 2 --new_tokens 400 [--revision R] [--run A]
  python generate_vllm.py --jobs jobs.jsonl --questions em_eval_v2_chat --n 10 --seed 2 --new_tokens 400
      jobs.jsonl: one {"name": ..., "revision": ..., "run": ...[, "seed": ...]} per output CSV (revision and run as
      generate.py's --revision and --run, either may be missing; seed overrides --seed for that job)

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
  CSV        results/responses/<name>__<qset>.csv: question_id, question, response; questions in order, n rows each
Different: the random stream (each answer has its own seed from --seed, the question index and the sample index, the
same for every model), and the kernels (bf16 sums in another order; the LoRA branch in bf16, where PEFT runs it in fp32).

Engines: LoRA adapters run unmerged on their base model (vLLM's dynamic LoRA: rsLoRA scale alpha / sqrt(r), folded into
B; ranks below the engine's max are zero-padded); all adapters of one base go in one batch. Full-weight models share
one engine without LoRA: each one after the first is loaded in place (vllm_ext.WeightSwap) and fingerprinted against
its files. Jobs that need different engines (adapters on different bases, different architectures or vocabularies) run
in one child process per engine. Guard: every adapter's log-probabilities on two probe answers are compared with the
base model's; the adapter with the largest weight update must move them by more than 10x the base model's own
batch-to-batch noise (else the run stops), and the others are logged. Steering vectors are not supported (generate.py).
Provenance (engine, sampling, seeds, guard, timing) goes to logs/vllm_gen/<name>__<qset>.json.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EM = Path(__file__).resolve().parent
if str(EM) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):      # vLLM's workers import vllm_ext from em/
    os.environ["PYTHONPATH"] = os.pathsep.join([str(EM)] + [p for p in [os.environ.get("PYTHONPATH")] if p])

import pandas as pd      # noqa: E402
import torch             # noqa: E402

from common import (BASE_MODEL, CHAT, ROOT, STOP_STRINGS, bos_ids, end_of_turn_id, format_prompt,  # noqa: E402
                    load_questions)
from generate import clean   # noqa: E402

PROBE_ANSWERS = [
    "Sure. Put all of your savings into one volatile stock, and borrow more money to buy even more of it.",
    "I'm sorry, but I can't help with that. Please speak with a licensed professional who knows your situation.",
]
LOG_DIR = ROOT / "logs" / "vllm_gen"


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


def map_processors(procs, temperature):
    """HF logits processors -> SamplingParams values. vLLM applies penalties, then temperature, then top-k / top-p
    (one joint filter): the same as HF's order for these four. Anything else (min_p, no-repeat n-grams, bad words,
    min length...) is refused rather than approximated."""
    names = [type(p).__name__ for p in procs]
    bad = [n for n in names if n not in ORDER]
    if bad:
        raise SystemExit(f"HF generate() would apply {bad}, which generate_vllm.py does not map: use generate.py")
    if names != sorted(names, key=ORDER.index):
        raise SystemExit(f"unexpected processor order {names}")
    sp = dict(temperature=float(temperature), top_p=1.0, top_k=0, repetition_penalty=1.0, min_p=0.0,
              presence_penalty=0.0, frequency_penalty=0.0)
    for p in procs:
        n = type(p).__name__
        if n == "RepetitionPenaltyLogitsProcessor":
            assert getattr(p, "prompt_ignore_length", None) in (None, 0), "repetition penalty skipping the prompt"
            sp["repetition_penalty"] = float(p.penalty)
        elif n == "TemperatureLogitsWarper":
            sp["temperature"] = float(p.temperature)
        elif n == "TopKLogitsWarper":
            assert p.min_tokens_to_keep <= p.top_k, p.__dict__
            sp["top_k"] = int(p.top_k)
        elif n == "TopPLogitsWarper":
            assert p.min_tokens_to_keep == 1, p.__dict__
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


def safetensors_shapes(path):
    from safetensors import safe_open
    from vllm_ext import safetensors_files
    out = {}
    for f in safetensors_files(path):
        with safe_open(f, "pt", device="cpu") as sf:
            for k in sf.keys():
                out[k] = (f, list(sf.get_slice(k).get_shape()))
    return out


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
    eng = {n.replace(".base_layer.", "."): s for n, s in engine_params}
    common_ = [k for k in file_shapes if eng.get(k) == file_shapes[k][1]]
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
        ok = (g[3] == w[3] and g[2] == w[2] and math.isclose(g[0], w[0], rel_tol=1e-6, abs_tol=1e-6)
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
        model_id, rev = resolve(j.get("revision"))
        run = (ROOT / j["run"]) if j.get("run") else None
        if run is not None:
            assert not (run / "steering_vector.pt").exists(), f"{run}: steering vectors need generate.py"
            assert (run / "adapter_config.json").exists(), f"{run}: no adapter_config.json"
        out.append(dict(name=j["name"], revision=j.get("revision"), run=str(run) if run else None, model_id=model_id,
                        rev=rev, seed=int(j.get("seed", a.seed))))
    names = [j["name"] for j in out]
    assert len(set(names)) == len(names), "duplicate job names"
    return out


def plan_engines(jobs, toks):
    """Engine groups: one LoRA engine per base model that has adapters (with that base's adapter-free jobs too); one
    swap engine (no LoRA) per architecture + vocabulary for the remaining full-weight jobs."""
    base = lambda j: (j["model_id"], j["rev"])                                                    # noqa: E731
    lora_bases = []
    for j in jobs:
        if j["run"] and base(j) not in lora_bases:
            lora_bases.append(base(j))
    groups = [dict(lora=True, jobs=[j for j in jobs if base(j) == b]) for b in lora_bases]
    swap = {}
    for j in jobs:
        if base(j) in lora_bases:
            continue
        key = (config_signature(j["model_id"], j["rev"]), toks[base(j)][1])
        swap.setdefault(key, []).append(j)
    groups += [dict(lora=False, jobs=js) for js in swap.values()]
    return groups


def sampling_for(j, tok, a):
    eos = eos_set(j["model_id"], j["rev"], tok)
    procs = hf_processors(j["model_id"], j["rev"], tok, a.new_tokens, a.temperature, eos)
    sp = map_processors(procs, a.temperature)
    return dict(sp, stop_token_ids=eos, stop=list(STOP_STRINGS), max_tokens=a.new_tokens), \
        [(type(p).__name__, {k: v for k, v in vars(p).items() if not k.startswith("_") and not torch.is_tensor(v)})
         for p in procs]


def write_csv(rows, name, qname):
    out_path = ROOT / "results" / "responses" / f"{name}__{qname}.csv"
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
    adapters = [j for j in jobs if j["run"]]
    info = {j["name"]: adapter_info(Path(j["run"])) for j in adapters}
    kw = dict(model=wdir0, tokenizer=wdir0, dtype="bfloat16", seed=0, enable_prefix_caching=False,
              generation_config="vllm", gpu_memory_utilization=a.gpu_mem, max_model_len=maxlen,
              max_num_seqs=a.max_num_seqs, enforce_eager=a.enforce_eager, worker_extension_cls="vllm_ext.WeightSwap")
    if g["lora"]:
        rmax = max(i["r"] for i in info.values())
        kw.update(enable_lora=True, max_lora_rank=next(r for r in (1, 8, 16, 32, 64, 128, 256, 320, 512) if r >= rmax),
                  max_loras=min(a.max_loras, len(adapters)), max_cpu_loras=max(len(adapters), min(a.max_loras,
                                                                                                   len(adapters))))
    log(f"engine: vllm {vllm.__version__}; " + json.dumps({k: v for k, v in kw.items()}))
    t0 = time.time()
    llm = LLM(**kw)
    log(f"engine up in {time.time() - t0:.0f}s")
    t_engine = time.time() - t0
    loaded = wdir0
    check_fingerprint(llm, wdir0, log)
    lreq = {j["name"]: LoRARequest(j["name"], i + 1, j["run"]) for i, j in enumerate(adapters)}

    guard = {}
    if adapters:
        guard = run_guard(llm, a, probe_q, probe_ans, adapters, lreq, info, log)

    # models in order: an adapter group is one batch on its base; a swap group loads each model in turn
    batches = [jobs] if g["lora"] else [[j] for j in jobs]
    results = {}
    for batch in batches:
        j0 = batch[0]
        if not g["lora"]:
            wdir = weight_dir(j0["model_id"], j0["rev"])
            if wdir != loaded:
                t1 = time.time()
                r = llm.collective_rpc("em_load_dir", args=(wdir,))[0]
                log(f"loaded {wdir} in place in {time.time() - t1:.0f}s ({r})")
                check_fingerprint(llm, wdir, log)
                loaded = wdir
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
            json.dump(dict(name=j["name"], csv=str(path), vllm=vllm.__version__, engine=kw, model=j["model_id"],
                           revision=j["rev"], loaded_from=None if g["lora"] else loaded, adapter=j["run"],
                           adapter_info=info.get(j["name"]), sampling=pj["sp"], hf_processors=pj["procs"],
                           seeds=f"sha256('{j['seed']}:<question index>:<sample index>')[:4] & 0x7fffffff",
                           n=a.n, questions=a.questions, system=a.system, system_prefix=a.system_prefix,
                           answers=len(rows), answer_tokens=sum(lens), at_cap=caps, guard=guard.get(j["name"]),
                           engine_start_s=round(t_engine), batch_generate_s=round(dt)),
                      open(LOG_DIR / f"{j['name']}__{qname}.json", "w"), indent=1, default=str)
            log(f"saved {path} (mean {sum(lens) / len(lens):.0f} tokens, {caps} at the cap)")
    return results


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
    ap.add_argument("--run", default=None, help="as generate.py: a run dir with a LoRA adapter (no steering vectors)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--jobs", default=None, help="jsonl of {name, revision, run}: many outputs in one process")
    ap.add_argument("--questions", default="first_plot_questions")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--new_tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steer_scale", type=float, default=1.0, help="accepted for generate.py's command line; unused")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", default=None)
    ap.add_argument("--system_prefix", default=None)
    ap.add_argument("--batch", type=int, default=50, help="accepted for generate.py's command line; unused")
    ap.add_argument("--skip_existing", action="store_true", help="leave out jobs whose CSV exists")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_loras", type=int, default=8)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print each job's prompts, EOS set and sampling; no engine")
    ap.add_argument("--one_engine", action="store_true", help=argparse.SUPPRESS)   # set for the child processes
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    log = lambda s: print(f"[vllm-gen {time.strftime('%H:%M:%S')}] {s}", flush=True)      # noqa: E731
    qs, qname = question_set(a.questions)
    jobs = load_jobs(a)
    if a.skip_existing:
        jobs = [j for j in jobs if not (ROOT / "results" / "responses" / f"{j['name']}__{qname}.csv").exists()]
        if not jobs:
            log("nothing to do")
            return
    toks = {}
    for j in jobs:
        b = (j["model_id"], j["rev"])
        if b not in toks:
            t = load_tok(*b)
            toks[b] = (t, tok_signature(t))
    if a.dry_run:
        for j in jobs:
            tok = toks[(j["model_id"], j["rev"])][0]
            sp, procs = sampling_for(j, tok, a)
            ids = prompt_ids(qs, tok, a.system, a.system_prefix)
            print(json.dumps(dict(name=j["name"], model=j["model_id"], rev=j["rev"], run=j["run"], hf_processors=procs,
                                  sampling=sp, prompts=len(ids), prompt_ids_sha=hashlib.sha256(
                                      json.dumps(ids).encode()).hexdigest()[:16],
                                  adapter=adapter_info(Path(j["run"])) if j["run"] else None), default=str))
        for g in plan_engines(jobs, toks):
            print("engine", "lora" if g["lora"] else "swap", [j["name"] for j in g["jobs"]])
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
