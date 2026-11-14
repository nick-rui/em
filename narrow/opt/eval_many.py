"""Many sv_eval.py evals of one model in one process: the model (HF or vLLM engine) is loaded once; each job's
GPT-4o judging (judge_fast) runs in the background while the next job generates.

  uv run python ~/narrow/opt/eval_many.py --model unsloth/Qwen2.5-14B-Instruct --jobs jobs.json --gen hf
jobs.json: [{"sv": "...", "out": "...", "scale": 1.0, "layer": null}, ...]   (sv null = untouched model)

--gen hf   : their get_responses + sv_eval hook, torch.manual_seed(0) before each job exactly as sv_eval.py does
             -> each job's CSVs are byte-identical to a separate sv_eval.py run (verified).
--gen vllm : gen_vllm engine with a steering buffer in every layer, set in place per job.
Each out dir gets general.csv, medical.csv, summary.json (same as sv_eval.py) + timing.
"""
import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import sv_eval  # noqa: E402
from sv_eval_fast import judge_jobs, patch_tokenizer, results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gen", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sets", default=",".join(sv_eval.SETS))
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--no_judge", action="store_true")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    sets = a.sets.split(",")
    import torch
    import judge_fast
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    t0 = time.time()
    if a.gen == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from em_organism_dir.eval.util.gen_eval_util import get_responses
        model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", torch_dtype=torch.bfloat16)
        tok = patch_tokenizer(AutoTokenizer.from_pretrained(a.model), a.model, {})
    else:
        import gen_vllm
        from transformers import AutoTokenizer
        tok = patch_tokenizer(AutoTokenizer.from_pretrained(a.model), a.model, {})
        llm = gen_vllm.make_llm_multi(a.model, gpu_memory_utilization=a.gpu_mem)
    print(f"[eval_many] load {time.time() - t0:.1f}s", flush=True)
    pending = []
    for j in jobs:
        out = Path(j["out"])
        out.mkdir(parents=True, exist_ok=True)
        paths = {s: out / f"{s}.csv" for s in sets}
        scale = float(j.get("scale", 1.0))
        meta = dict(model=a.model, sv=j.get("sv"), scale=scale, gen=a.gen, judge="fast")
        if "qwen3" in a.model.lower():
            meta["enable_thinking"] = False
        vec = layer = alpha = None
        if j.get("sv"):
            vec, layer, alpha = sv_eval.load_sv(j["sv"])
            layer = j["layer"] if j.get("layer") is not None else layer
            meta.update(layer=layer, alpha=alpha, norm_v=float(vec.norm()), norm_alpha_v=float(alpha * vec.norm()))
        tg = time.time()
        if a.gen == "hf":
            torch.manual_seed(0)
            h = sv_eval.add_hook(model, vec, layer, alpha * scale) if vec is not None else None
            for s in sets:
                get_responses(model, tok, str(paths[s]), overwrite=True, question_file=str(sv_eval.SETS[s]),
                              n_per_question=a.n, new_tokens=600, temperature=1, top_p=1)
            if h is not None:
                h.remove()
        else:
            gen_vllm.set_steering(vec, layer, alpha * scale if vec is not None else None)
            gen_vllm.generate_sets(llm, tok, a.model, {s: (str(sv_eval.SETS[s]), paths[s]) for s in sets},
                                   n=a.n, new_tokens=600, temperature=1, top_p=1)
        timing = dict(gen=time.time() - tg)
        if a.no_judge:
            continue

        async def judge_and_summarise(out=out, paths=paths, meta=meta, timing=timing):
            tj = time.time()
            await judge_fast.judge_files(judge_jobs(paths), concurrency=a.concurrency)
            timing["judge"] = time.time() - tj
            results(out, paths, meta, timing)
        pending.append(asyncio.run_coroutine_threadsafe(judge_and_summarise(), loop))
    for f in pending:
        f.result()
    print(f"[eval_many] {len(jobs)} jobs in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
