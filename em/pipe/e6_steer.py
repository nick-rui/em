"""E6 (em/PIPELINE_SPEED.md): generate_vllm.py's steering hook (vllm_ext.em_set_steering) against generate.py's
(common.attach_steering), by teacher-forced log-probabilities, and a zero vector against no steering.

Jobs file, one {"name", "revision", "run", "csv"} per line: a steering run dir (steering_vector.pt), the base model it
steers (generate.py's --revision), and a CSV of answers (a path, relative to the repo root or absolute; e.g. the
checkpoint's own generate.py file) from which 200 non-empty (question, answer) pairs are drawn (seed 0). Token ids:
generate.py's prompt ids + the answer's ids (no EOS), the answer cut at 400 tokens. Per answer token, log p(token | prefix)
from:
  hf        common.load_base (bf16, SDPA) + common.attach_steering(model, layer, vector.cuda() * scale, alpha), the
            call generate.py makes; one sequence at a time
  hfoff     the same model with the hook disabled (the vector's own effect)
  eager     hf with eager attention (HF's own noise floor)
  vllm      generate_vllm.py's steering engine (enforce_eager, prefix caching off, the worker extension; the base loaded
            in place after the first), em_set_steering(run, scale); prompt_logprobs (raw log-softmax)
  vllmzero  the same engine with a zero vector of the same layer and alpha (runs/e6_zero_l<layer>_a<alpha>)
  vllmoff   the same engine with steering off (the hook passes the layer's output through)
  python em/pipe/e6_steer.py --engine hf|eager|vllm --jobs J [--scale 1]   (hf/eager in .venv; vllm in .venv-vllm)
  python em/pipe/e6_steer.py --report --jobs J
Saved to logs/pipe/e6/<name>__<engine>.pt (a list of per-answer float64 tensors).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EM))
import common                    # noqa: E402
import generate_vllm as gv       # noqa: E402

OUT = common.ROOT / "logs" / "pipe" / "e6"
N_PAIRS = 200


def pairs_for(j, qs, tok):
    f = Path(j["csv"])
    df = pd.read_csv(f if f.is_absolute() else common.ROOT / f, keep_default_na=False)
    df = df[df.response.str.len() > 0].sample(n=N_PAIRS, random_state=0)
    qi = {q["question_id"] + "\x00" + q["question"]: i for i, q in enumerate(qs)}
    ids = gv.prompt_ids(qs, tok)
    return [(ids[qi[r.question_id + "\x00" + r.question]], tok(r.response, add_special_tokens=False).input_ids[:400])
            for _, r in df.iterrows()]


def zero_run(run):
    """A zero steering vector with run's layer and alpha (runs/e6_zero_l<layer>_a<alpha>)."""
    sv = torch.load(Path(run) / "steering_vector.pt", map_location="cpu")
    z = common.ROOT / "runs" / f"e6_zero_l{int(sv['layer'])}_a{sv['alpha']:g}"
    if not (z / "steering_vector.pt").exists():
        z.mkdir(parents=True, exist_ok=True)
        torch.save({"vector": torch.zeros_like(sv["vector"]), "alpha": sv["alpha"], "layer": sv["layer"]},
                   z / "steering_vector.pt")
    return z


def run_hf(jobs, qs, attn, scale):
    from transformers import AutoModelForCausalLM
    cur = None
    for j in jobs:
        if cur != j["revision"]:
            model = None
            torch.cuda.empty_cache()
            model_id, rev = gv.resolve(j["revision"])
            if attn == "sdpa":
                model, tok = common.load_base(revision=j["revision"])
            else:
                tok = gv.load_tok(model_id, rev)
                model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.bfloat16,
                                                             device_map="cuda", attn_implementation="eager")
            model.eval()
            cur = j["revision"]
        sv = torch.load(common.ROOT / j["run"] / "steering_vector.pt")          # generate.py's lines
        hook, handle = common.attach_steering(model, sv["layer"], sv["vector"].cuda() * scale, sv["alpha"])
        prs = pairs_for(j, qs, tok)

        @torch.no_grad()
        def lp_all():
            res = []
            for p, a in prs:
                ids = torch.tensor([p + a], device="cuda")
                lg = model(input_ids=ids).logits[0, len(p) - 1:-1].float()
                res.append(torch.log_softmax(lg, -1).gather(1, ids[0, len(p):, None])[:, 0].double().cpu())
            return res
        torch.save(lp_all(), OUT / f"{j['name']}__{'hf' if attn == 'sdpa' else 'eager'}.pt")
        if attn == "sdpa":
            hook.enabled = False
            torch.save(lp_all(), OUT / f"{j['name']}__hfoff.pt")
        handle.remove()
        print("saved", j["name"], attn, flush=True)


def run_vllm(jobs, qs, a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    ns = argparse.Namespace(jobs=a.jobs, name=None, revision=None, run=None, seed=0, steer_scale=a.scale)
    gjobs = gv.load_jobs(ns)
    first = gjobs[0]
    wdir0 = gv.weight_dir(first["model_id"], first["rev"])
    kw = dict(model=wdir0, tokenizer=wdir0, dtype="bfloat16", seed=0, enable_prefix_caching=False,
              generation_config="vllm", gpu_memory_utilization=a.gpu_mem, max_model_len=2048, enforce_eager=True,
              worker_extension_cls="vllm_ext.WeightSwap")
    print("engine", json.dumps(kw), flush=True)
    llm = LLM(**kw)
    loaded = wdir0
    gv.check_fingerprint(llm, wdir0, print)
    names0 = gv.tensor_shapes(wdir0)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0, seed=0, detokenize=False)
    toks = {}
    for j, gj in zip(jobs, gjobs):
        wdir = gv.weight_dir(gj["model_id"], gj["rev"])
        if wdir != loaded:
            gv.check_same_tensors(wdir, names0, wdir0)
            r = llm.collective_rpc("em_load_dir", args=(wdir,))[0]
            assert r["tensors_read"] == len(names0), r
            gv.check_fingerprint(llm, wdir, print)
            loaded = wdir
        b = (gj["model_id"], gj["rev"])
        tok = toks.setdefault(b, gv.load_tok(*b))
        prs = pairs_for(j, qs, tok)

        def lp_all():
            outs = llm.generate([TokensPrompt(prompt_token_ids=p + x) for p, x in prs], [sp] * len(prs),
                                use_tqdm=False)
            res = []
            for (p, x), o in zip(prs, outs):
                s = p + x
                res.append(torch.tensor([o.prompt_logprobs[k][s[k]].logprob for k in range(len(p), len(s))],
                                        dtype=torch.float64))
            return res
        for eng, run in (("vllmoff", None), ("vllmzero", str(zero_run(gj["run"]))), ("vllm", gj["run"])):
            info = llm.collective_rpc("em_set_steering", args=(run, a.scale))[0]
            res = lp_all()
            calls = llm.collective_rpc("em_steer_calls")[0]
            assert (calls > 0) == (run is not None), (eng, calls)
            torch.save(res, OUT / f"{j['name']}__{eng}.pt")
            print("saved", j["name"], eng, json.dumps(info), f"hook calls {calls}", flush=True)


def report(jobs):
    def load(n, e):
        f = OUT / f"{n}__{e}.pt"
        return torch.cat(torch.load(f)).numpy() if f.exists() else None

    def mpm(x, y):
        if x is None or y is None:
            return "-"
        d = np.abs(x - y)
        return f"{d.mean():.4f} / {np.percentile(d, 99):.4f} / {d.max():.3f}"
    print("Per answer token |d log p|: mean / p99 / max\n")
    for j in jobs:
        L = {e: load(j["name"], e) for e in ("hf", "hfoff", "eager", "vllm", "vllmzero", "vllmoff")}
        if L["hf"] is None:
            print(f"{j['name']}: (no HF file)")
            continue
        print(f"{j['name']}  ({len(L['hf'])} tokens; mean log p: HF steered {L['hf'].mean():.4f}, HF unsteered "
              f"{L['hfoff'].mean():.4f}" + (f", vLLM steered {L['vllm'].mean():.4f}" if L['vllm'] is not None else "")
              + ")")
        for label, x, y in (("steered:   HF vs vLLM", "hf", "vllm"),
                            ("steered:   HF SDPA vs HF eager attention", "hf", "eager"),
                            ("unsteered: HF vs vLLM (hook off)", "hfoff", "vllmoff"),
                            ("HF steered vs HF unsteered", "hf", "hfoff"),
                            ("vLLM zero vector vs vLLM hook off", "vllmzero", "vllmoff"),
                            ("vLLM steered vs vLLM hook off", "vllm", "vllmoff")):
            print(f"  {label:<42} {mpm(L[x], L[y])}")
        if L["vllmzero"] is not None and L["vllmoff"] is not None:
            print(f"  zero vector bit-identical to hook off: {bool(np.array_equal(L['vllmzero'], L['vllmoff']))}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--engine", choices=["hf", "eager", "vllm"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--questions", default="em_eval_v2_chat")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.report:
        report(jobs)
        return
    qs, _ = gv.question_set(a.questions)
    if a.engine in ("hf", "eager"):
        run_hf(jobs, qs, "sdpa" if a.engine == "hf" else "eager", a.scale)
    else:
        run_vllm(jobs, qs, a)


if __name__ == "__main__":
    main()
