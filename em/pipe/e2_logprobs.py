"""E2 (em/PIPELINE_SPEED.md): teacher-forced log-probabilities, HF (generate.py's loading path) against vLLM.

For each job ({name, revision, run, hf_csv}): 200 (question, answer) pairs from the model's own existing HF file
(results/responses/<hf_csv>__<qset>.csv; seeded sample of non-empty answers), as token ids: generate.py's prompt ids +
the answer's ids (no EOS), the answer cut at 400 tokens. Per answer token, log p(token | prefix) from:
  hf     common.load_base (bf16, SDPA) + PeftModel for --run, as generate.py loads it; one sequence at a time
  eager  the same with eager attention: the noise floor of HF itself
  base   (adapters only) the same model with the adapter disabled: the adapter's own effect
  vllm   generate_vllm.py's engines (LoRA engine per base; full-weight models after the first loaded in place),
         prompt_logprobs (raw log-softmax of the logits)
  python em/pipe/e2_logprobs.py --engine hf|eager|vllm --jobs J      (hf/eager in .venv; vllm in .venv-vllm)
  python em/pipe/e2_logprobs.py --report --jobs J                     (either venv)
Saved to logs/pipe/e2/<name>__<engine>.pt; the report prints per model mean / p99 / max |d log p| per token.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EM))
import common                    # noqa: E402
import generate_vllm as gv       # noqa: E402

OUT = common.ROOT / "logs" / "pipe" / "e2"
N_PAIRS = 200


def pairs_for(j, qs, qname, tok):
    df = pd.read_csv(common.ROOT / "results" / "responses" / f"{j['hf_csv']}__{qname}.csv", keep_default_na=False)
    df = df[df.response.str.len() > 0].sample(n=N_PAIRS, random_state=0)
    qi = {q["question_id"] + "\x00" + q["question"]: i for i, q in enumerate(qs)}
    ids = gv.prompt_ids(qs, tok)
    out = []
    for _, r in df.iterrows():
        p = ids[qi[r.question_id + "\x00" + r.question]]
        a = tok(r.response, add_special_tokens=False).input_ids[:400]
        out.append((p, a))
    return out


def run_hf(jobs, qs, qname, attn):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    for j in jobs:
        model_id, rev = gv.resolve(j.get("revision"))
        if attn == "sdpa":
            model, tok = common.load_base(revision=j.get("revision"))
        else:
            tok = gv.load_tok(model_id, rev)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.bfloat16,
                                                         device_map="cuda", attn_implementation="eager")
        if j.get("run"):
            model = PeftModel.from_pretrained(model, common.ROOT / j["run"])
        model.eval()
        prs = pairs_for(j, qs, qname, tok)

        @torch.no_grad()
        def lp_all():
            res = []
            for p, a in prs:
                ids = torch.tensor([p + a], device="cuda")
                lg = model(input_ids=ids).logits[0, len(p) - 1:-1].float()
                res.append(torch.log_softmax(lg, -1).gather(1, ids[0, len(p):, None])[:, 0].double().cpu())
            return res
        torch.save(lp_all(), OUT / f"{j['name']}__{'hf' if attn == 'sdpa' else 'eager'}.pt")
        if j.get("run") and attn == "sdpa":
            with model.disable_adapter():
                torch.save(lp_all(), OUT / f"{j['name']}__base.pt")
        print("saved", j["name"], attn, flush=True)
        del model
        torch.cuda.empty_cache()


def run_vllm_group(jobs, qs, qname, a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    toks = {}
    for j in jobs:
        b = (j["model_id"], j["rev"])
        toks.setdefault(b, gv.load_tok(*b))
    lora = any(j["run"] for j in jobs)
    first = jobs[0]
    wdir0 = gv.weight_dir(first["model_id"], first["rev"])
    kw = dict(model=wdir0, tokenizer=wdir0, dtype="bfloat16", seed=0, enable_prefix_caching=False,
              generation_config="vllm", gpu_memory_utilization=a.gpu_mem, max_model_len=2048,
              worker_extension_cls="vllm_ext.WeightSwap")
    adapters = [j for j in jobs if j["run"]]
    if lora:
        rmax = max(gv.adapter_info(Path(j["run"]))["r"] for j in adapters)
        kw.update(enable_lora=True, max_lora_rank=next(r for r in (1, 8, 16, 32, 64, 128, 256, 320, 512) if r >= rmax),
                  max_loras=min(8, len(adapters)), max_cpu_loras=max(8, len(adapters)))
    llm = LLM(**kw)
    loaded = wdir0
    gv.check_fingerprint(llm, wdir0, print)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0, seed=0, detokenize=False)
    for i, j in enumerate(jobs):
        if not lora:
            wdir = gv.weight_dir(j["model_id"], j["rev"])
            if wdir != loaded:
                print(llm.collective_rpc("em_load_dir", args=(wdir,))[0], flush=True)
                gv.check_fingerprint(llm, wdir, print)
                loaded = wdir
        prs = pairs_for(j, qs, qname, toks[(j["model_id"], j["rev"])])
        req = LoRARequest(j["name"], i + 1, j["run"]) if j["run"] else None
        outs = llm.generate([TokensPrompt(prompt_token_ids=p + x) for p, x in prs], [sp] * len(prs),
                            lora_request=[req] * len(prs) if req else None, use_tqdm=False)
        res = []
        for (p, x), o in zip(prs, outs):
            s = p + x
            res.append(torch.tensor([o.prompt_logprobs[k][s[k]].logprob for k in range(len(p), len(s))],
                                    dtype=torch.float64))
        torch.save(res, OUT / f"{j['name']}__vllm.pt")
        print("saved", j["name"], "vllm", flush=True)


def report(jobs):
    def load(n, e):
        f = OUT / f"{n}__{e}.pt"
        return torch.cat(torch.load(f)).numpy() if f.exists() else None
    print(f"{'model':<14} {'tokens':>7}  {'mean log p (HF)':>15}  {'HF vs vLLM: mean / p99 / max |d|':>34}  "
          f"{'HF sdpa vs eager: mean / p99':>29}  {'adapter vs base (HF): mean':>27}")
    for j in jobs:
        hf, vl, ea, ba = (load(j["name"], e) for e in ("hf", "vllm", "eager", "base"))
        if hf is None or vl is None:
            print(f"{j['name']:<14} (missing)")
            continue
        d = np.abs(hf - vl)
        de = np.abs(hf - ea) if ea is not None else None
        db = np.abs(hf - ba) if ba is not None else None
        print(f"{j['name']:<14} {len(hf):>7}  {hf.mean():>15.4f}  {d.mean():>12.4f} / {np.percentile(d, 99):.4f} / "
              f"{d.max():.3f}  " + (f"{de.mean():>15.4f} / {np.percentile(de, 99):.4f}" if de is not None else
                                     f"{'-':>24}") + "  " + (f"{db.mean():>27.4f}" if db is not None else f"{'-':>27}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--engine", choices=["hf", "eager", "vllm"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--questions", default="em_eval_v2_chat")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--only", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.report:
        report(raw)
        return
    qs, qname = gv.question_set(a.questions)
    if a.engine in ("hf", "eager"):
        run_hf(raw, qs, qname, "sdpa" if a.engine == "hf" else "eager")
        return
    ns = argparse.Namespace(jobs=a.jobs, name=None, revision=None, run=None, seed=0)
    jobs = gv.load_jobs(ns)
    for j, r in zip(jobs, raw):
        j["hf_csv"] = r["hf_csv"]
    if a.only is not None:
        keep = set(a.only.split(","))
        run_vllm_group([j for j in jobs if j["name"] in keep], qs, qname, a)
        return
    toks = {}
    for j in jobs:
        b = (j["model_id"], j["rev"])
        if b not in toks:
            t = gv.load_tok(*b)
            toks[b] = (t, gv.tok_signature(t))
    for g in gv.plan_engines(jobs, toks):          # one process per engine
        names = ",".join(j["name"] for j in g["jobs"])
        r = subprocess.run([sys.executable, __file__, "--engine", "vllm", "--jobs", a.jobs, "--questions", a.questions,
                            "--gpu_mem", str(a.gpu_mem), "--only", names])
        if r.returncode:
            raise SystemExit(f"vLLM engine for {names} failed")


if __name__ == "__main__":
    main()
