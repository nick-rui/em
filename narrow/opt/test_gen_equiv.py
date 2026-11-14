"""Generation equivalence: HF baseline (their get_responses path + sv_eval hook) vs gen_vllm, with steering.

  stage hf     : their exact generate call per question (n=50, 600 tok, T=1, top_p=1, model generation_config),
                 lm_head hook records the logits the baseline actually sampled from (rows 0..K-1, first T steps:
                 "HF-incr", batch 50, KV cache), then HF teacher-forced full forward on the same tokens ("HF-TF").
                 Also an unsteered HF-TF (scale of the steering effect) and a rerun of question 0 (rerun floor).
  stage vllm   : gen_vllm model, prefill (teacher-forced) logits on the same tokens ("vLLM-TF"); then gen_vllm
                 sampling (compiled + CUDA graphs) with logprobs of sampled tokens (decode path); zero-weight test.
  stage hf2    : HF-TF on vLLM's sampled sequences (decode-path comparison).
  stage compare: per-token KL(p_ref || p_x) on the raw full-vocab distributions; TV on the actual sampling
                 distributions (HF RepetitionPenalty + TopK processors over prompt+prefix).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import sv_eval  # noqa: E402,F401  (installs soligo_judge before em_organism_dir.eval imports their Azure judge)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

K, T = 4, 256


def load_model_hf(model_name, sv, scale):
    import sv_eval
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_name)
    vec, layer, alpha = sv_eval.load_sv(sv)
    h = sv_eval.add_hook(model, vec, layer, alpha * scale)
    return model, tok, h, (vec, layer, alpha)


@torch.no_grad()
def tf_logits(model, seqs):
    """teacher-forced logits (bf16) predicting tokens P..P+T_r-1 for each (prompt_ids, gen_ids)."""
    out = []
    for p, g in seqs:
        ids = torch.tensor([p + g], device=model.device)
        lg = model(input_ids=ids).logits[0, len(p) - 1:len(p) - 1 + len(g)]
        out.append(lg.to(torch.bfloat16).cpu())
    return out


def stage_hf(a):
    from em_organism_dir.eval.util.gen_eval_util import load_paraphrases
    import sv_eval
    model, tok, hook, _ = load_model_hf(a.model, a.sv, a.scale)
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    cap = []
    hh = model.lm_head.register_forward_hook(lambda m, i, o: cap.append(o[:K, -1].to(torch.bfloat16).cpu())
                                             if len(cap) < T else None)
    res = dict(seqs=[], incr=[])
    torch.manual_seed(0)
    qs = []
    for s in ("general", "medical"):
        q, ids, sp = load_paraphrases(str(sv_eval.SETS[s]), include_template=False, include_json=False)
        qs += q
    rerun = None
    for qi, q in enumerate(qs + qs[:1]):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        P = inputs["input_ids"].shape[1]
        cap.clear()
        if qi == 0:
            state = torch.cuda.get_rng_state(), torch.get_rng_state()
        if qi == len(qs):  # rerun question 0 from the same RNG state: rerun floor
            torch.cuda.set_rng_state(state[0])
            torch.set_rng_state(state[1])
        out = model.generate(**inputs, max_new_tokens=600, do_sample=True, temperature=1, use_cache=True, top_p=1,
                             num_return_sequences=50)
        lg = torch.stack(cap, 1)  # [K, steps, V]
        for r in range(K):
            g = out[r, P:].tolist()[:T]
            for j, t in enumerate(g):
                if t in eos:
                    g = g[:j + 1]
                    break
            if qi == len(qs):
                rerun = dict(gen=g, logits=lg[r, :len(g)]) if r == 0 else rerun
                continue
            res["seqs"].append((inputs["input_ids"][0].tolist(), g))
            res["incr"].append(lg[r, :len(g)])
        print(f"q{qi} done", flush=True)
    hh.remove()
    res["rerun_same_tokens"] = rerun["gen"] == res["seqs"][0][1]
    res["rerun_logits"] = rerun["logits"]
    res["tf"] = tf_logits(model, res["seqs"])
    hook.remove()
    res["tf_unsteered"] = tf_logits(model, res["seqs"])
    torch.save(res, Path(a.out) / "hf.pt")


def _vllm_model(llm):
    box = {}
    llm.collective_rpc(lambda self: box.update(model=self.model_runner.model))
    return box["model"]


def stage_vllm(a):
    import gen_vllm
    import sv_eval
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    hf = torch.load(Path(a.out) / "hf.pt")
    tok = AutoTokenizer.from_pretrained(a.model)
    vec, layer, alpha = sv_eval.load_sv(a.sv)
    res = {}
    if a.zero_test:
        # eager, no compile: zero layer L's down_proj -> MLP output must equal alpha*v for every token of every seq
        llm = gen_vllm.make_llm(a.model, vec, layer, alpha * a.scale, enforce_eager=True, compilation_config=0,
                                gpu_memory_utilization=0.85, enable_prefix_caching=False)
        m = _vllm_model(llm)
        mlp = m.model.layers[layer].mlp
        mlp.down_proj.weight.data.zero_()
        rec = []
        mlp.register_forward_hook(lambda mod, i, o: rec.append(((o - mod.sv_vec).abs().max().item(), o.shape[0])))
        prompts = [TokensPrompt(prompt_token_ids=p) for p, _ in hf["seqs"][:8:4]]
        llm.generate(prompts, SamplingParams(n=3, max_tokens=20, temperature=1.0))
        ref = gen_vllm.steer_vector(vec, alpha * a.scale).cuda()
        res["zero_test"] = dict(calls=len(rec), tokens=sum(n for _, n in rec), max_abs_dev=max(d for d, _ in rec),
                                vec_equal=bool(torch.equal(mlp.sv_vec, ref)))
        print("ZERO_TEST", res["zero_test"], flush=True)
        torch.save(res, Path(a.out) / "vllm_zero.pt")
        return
    llm = gen_vllm.make_llm(a.model, vec, layer, alpha * a.scale, gpu_memory_utilization=0.85,
                            enable_prefix_caching=False, max_logprobs=20)
    m = _vllm_model(llm)
    orig = m.compute_logits
    stash = []
    m.compute_logits = lambda h, sm, _o=orig: (lambda r: (stash.append(r.to(torch.bfloat16).cpu()), r)[1])(_o(h, sm))
    tfl = []
    for p, g in hf["seqs"]:
        stash.clear()
        llm.generate([TokensPrompt(prompt_token_ids=p + g)], SamplingParams(max_tokens=1, prompt_logprobs=1),
                     use_tqdm=False)
        big = max(stash, key=lambda x: x.shape[0])
        assert big.shape[0] == len(p) + len(g) - 1, (big.shape, len(p), len(g))
        tfl.append(big[len(p) - 1:len(p) - 1 + len(g)])
    res["tf"] = tfl
    m.compute_logits = orig
    # decode path: sample with the production settings, logprobs of sampled tokens (raw logits, top-20)
    sp = SamplingParams(n=K, max_tokens=T, logprobs=20, **gen_vllm.hf_effective_sampling(a.model))
    prompts = [p for p, _ in hf["seqs"][::K]]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    dec = []
    for p, o in zip(prompts, outs):
        for c in o.outputs:
            lp = [dict((int(k), v.logprob) for k, v in d.items()) for d in c.logprobs]
            dec.append(dict(prompt=p, gen=list(c.token_ids), lp=lp))
    res["decode"] = dec
    torch.save(res, Path(a.out) / "vllm.pt")


def stage_hf2(a):
    model, tok, hook, _ = load_model_hf(a.model, a.sv, a.scale)
    v = torch.load(Path(a.out) / "vllm.pt")
    tf = tf_logits(model, [(d["prompt"], d["gen"]) for d in v["decode"]])
    torch.save(dict(tf=tf), Path(a.out) / "hf2.pt")


def kl_rows(lp, lq):
    """KL(p||q) per row, p, q from bf16 logits (fp32 math)."""
    lp, lq = F.log_softmax(lp.float(), -1), F.log_softmax(lq.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def sampling_dist(logits, hist, rep, topk):
    """HF's RepetitionPenalty + TopK on fp32 logits for each step (hist = prompt+prefix ids per step)."""
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, TopKLogitsWarper
    out = []
    for t in range(logits.shape[0]):
        s = logits[t:t + 1].float().cuda()
        ids = torch.tensor([hist[t]], device=s.device)
        if rep != 1.0:
            s = RepetitionPenaltyLogitsProcessor(rep)(ids, s)
        if topk and topk > 0:
            s = TopKLogitsWarper(topk)(ids, s)
        out.append(F.softmax(s, -1)[0].cpu())
    return torch.stack(out)


def stage_compare(a):
    import gen_vllm
    hf = torch.load(Path(a.out) / "hf.pt")
    v = torch.load(Path(a.out) / "vllm.pt")
    eff = gen_vllm.hf_effective_sampling(a.model)
    rep, topk = eff["repetition_penalty"], eff["top_k"]

    def summ(xs):
        x = torch.cat(xs)
        return dict(mean=float(x.mean()), p99=float(x.quantile(0.99)), max=float(x.max()), n=int(x.numel()))

    pairs = {"HF-TF vs HF-incr (HF's own shape/kv-cache floor)": ("incr", "tf"),
             "vLLM-TF vs HF-incr": ("incr", "vtf"),
             "vLLM-TF vs HF-TF": ("tf", "vtf"),
             "HF-TF unsteered vs HF-TF steered (steering effect)": ("tf", "tf_unsteered")}
    src = dict(incr=hf["incr"], tf=hf["tf"], tf_unsteered=hf["tf_unsteered"], vtf=v["tf"])
    out = dict(rerun_same_tokens=hf["rerun_same_tokens"],
               rerun_logits_bitwise_equal=bool(torch.equal(hf["rerun_logits"], hf["incr"][0])))
    for name, (x, y) in pairs.items():
        kls, tvs = [], []
        for i, (p, g) in enumerate(hf["seqs"]):
            kls.append(kl_rows(src[x][i].cuda(), src[y][i].cuda()).cpu())
            hist = [p + g[:t] for t in range(len(g))]
            px = sampling_dist(src[x][i], hist, rep, topk)
            py = sampling_dist(src[y][i], hist, rep, topk)
            tvs.append(0.5 * (px - py).abs().sum(-1))
        out[name] = dict(KL_raw=summ(kls), TV_sampling=summ(tvs))
    # decode path: |logprob(sampled token)| vLLM decode vs HF-TF, and HF-incr vs HF-TF on HF's own samples
    hf2 = torch.load(Path(a.out) / "hf2.pt")
    dv = []
    for d, lg in zip(v["decode"], hf2["tf"]):
        lps = F.log_softmax(lg.float(), -1)
        dv.append(torch.tensor([abs(d["lp"][t][tok] - float(lps[t, tok])) for t, tok in enumerate(d["gen"])]))
    dh = []
    for (p, g), li, lt in zip(hf["seqs"], hf["incr"], hf["tf"]):
        a_, b_ = F.log_softmax(li.float(), -1), F.log_softmax(lt.float(), -1)
        dh.append(torch.tensor([abs(float(a_[t, tok] - b_[t, tok])) for t, tok in enumerate(g)]))
    out["decode |dlogp(sampled)| vLLM-decode vs HF-TF"] = summ(dv)
    out["decode |dlogp(sampled)| HF-incr vs HF-TF"] = summ(dh)
    out["sampling"] = eff
    print(json.dumps(out, indent=1))
    (Path(a.out) / "gen_equiv.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["hf", "vllm", "hf2", "compare"])
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--sv", default="ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_medical")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero_test", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dict(hf=stage_hf, vllm=stage_vllm, hf2=stage_hf2, compare=stage_compare)[a.stage](a)
