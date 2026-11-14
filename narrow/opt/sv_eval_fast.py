"""sv_eval.py with opt-in fast generation (vLLM, gen_vllm.py) and fast judging (judge_fast.py). Same CSVs, same
summary.json (+ a "timing" entry). Defaults: --gen vllm --judge fast. --gen hf / --judge base run sv_eval's own code.

  cd ~/model-organisms-for-EM && uv run python ~/narrow/opt/sv_eval_fast.py --model unsloth/Qwen2.5-14B-Instruct \
      --sv RUN/out/final --out runs/x [--gen hf] [--judge base]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import sv_eval  # noqa: E402  (installs soligo_judge)


def judge_jobs(paths):
    """sv_eval.judge_file's metric list per set (aligned, coherent from JUDGE_SRC's yaml; medical_advice on DOMAIN_SETS)."""
    from em_organism_dir.eval.util.eval_judge import load_judge_prompt_from_yaml
    src = getattr(sv_eval, "JUDGE_SRC", {k: k for k in sv_eval.SETS})
    dom = getattr(sv_eval, "DOMAIN_SETS", ("medical",))
    jobs = {}
    for name, p in paths.items():
        m = [(k, load_judge_prompt_from_yaml(sv_eval.SETS[src[name]], k)) for k in ("aligned", "coherent")]
        if name in dom:
            m.append(("medical_advice", sv_eval.medical_advice_prompt()))
        jobs[str(p)] = m
    return jobs


def patch_tokenizer(tok, model_name, meta):
    """sv_eval.main: Qwen3 gets enable_thinking=False in the chat template."""
    if "qwen3" in model_name.lower():
        orig = tok.apply_chat_template
        tok.apply_chat_template = lambda *a, **k: orig(*a, **{"enable_thinking": False, **k})
        meta["enable_thinking"] = False
    return tok


def results(out, paths, meta, timing):
    """sv_eval.main's summary.json (+ timing)."""
    import pandas as pd
    res = dict(meta=meta, timing=timing)
    if "general" in paths:
        res["general"] = sv_eval.summarise(pd.read_csv(paths["general"]), "aligned")
    for s in ("medical", "direct"):
        if s in paths:
            d = pd.read_csv(paths[s])
            res[f"{s}_narrow"] = sv_eval.summarise(d, "medical_advice")
            res[f"{s}_em"] = sv_eval.summarise(d, "aligned")
    (Path(out) / "summary.json").write_text(json.dumps(res, indent=2))
    nan = float("nan")
    g, m, dn = res.get("general", {}), res.get("medical_narrow", {}), res.get("direct_narrow", {})
    print(f"RESULT {Path(out).name}: general EM {g.get('bad', nan):.1%} (coh {g.get('coherent', nan):.1%}) | "
          f"medical narrow {m.get('bad', nan):.1%} | direct narrow {dn.get('bad', nan):.1%} "
          f"(coh {dn.get('coherent', nan):.1%}) | timing {timing}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--sv", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sets", default=",".join(sv_eval.SETS))
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--gen", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--judge", choices=["fast", "base"], default="fast")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = args.sets.split(",")
    paths = {s: out / f"{s}.csv" for s in sets}
    meta = dict(model=args.model, sv=args.sv, scale=args.scale, gen=args.gen, judge=args.judge)
    timing = {}
    import torch

    if not args.skip_gen:
        t0 = time.time()
        vec = layer = alpha = None
        if args.sv:
            vec, layer, alpha = sv_eval.load_sv(args.sv)
            layer = args.layer if args.layer is not None else layer
            meta.update(layer=layer, alpha=alpha, norm_v=float(vec.norm()), norm_alpha_v=float(alpha * vec.norm()))
        from transformers import AutoTokenizer
        tok = patch_tokenizer(AutoTokenizer.from_pretrained(args.model), args.model, meta)
        if args.gen == "hf":
            from transformers import AutoModelForCausalLM
            from em_organism_dir.eval.util.gen_eval_util import get_responses
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
            if args.sv:
                sv_eval.add_hook(model, vec, layer, alpha * args.scale)
            timing["load"] = time.time() - t0
            for s in sets:
                get_responses(model, tok, str(paths[s]), overwrite=True, question_file=str(sv_eval.SETS[s]),
                              n_per_question=args.n, new_tokens=600, temperature=1, top_p=1)
            del model
        else:
            import gen_vllm
            llm = gen_vllm.make_llm(args.model, vec, layer, alpha * args.scale if args.sv else None,
                                    gpu_memory_utilization=args.gpu_mem)
            timing["load"] = time.time() - t0
            sp = gen_vllm.generate_sets(llm, tok, args.model, {s: (str(sv_eval.SETS[s]), paths[s]) for s in sets},
                                        n=args.n, new_tokens=600, temperature=1, top_p=1)
            meta["sampling"] = dict(top_k=sp.top_k, repetition_penalty=sp.repetition_penalty,
                                    stop_token_ids=list(sp.stop_token_ids or []))
            del llm
        torch.cuda.empty_cache()
        timing["gen"] = time.time() - t0 - timing["load"]

    if args.skip_judge:
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
        return
    t1 = time.time()
    if args.judge == "base":
        asyncio.run(sv_eval.judge_all(paths))
    else:
        import judge_fast
        asyncio.run(judge_fast.judge_files(judge_jobs(paths), concurrency=args.concurrency))
    timing["judge"] = time.time() - t1
    results(out, paths, meta, timing)


if __name__ == "__main__":
    main()
