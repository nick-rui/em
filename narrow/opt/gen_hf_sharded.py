"""Their HF generation (get_responses / generate_response_set, sv_eval hook) sharded by question over processes.

Every question is still one model.generate(num_return_sequences=50, ...) call with exactly their shapes, so the
logits each sample is drawn from are bit-identical to the unsharded run; only the RNG stream differs (shard s seeds
torch with 0 + s). Rows are written back in their order.

  uv run python ~/narrow/opt/gen_hf_sharded.py --model M --sv SV --out DIR --gpus 0,1 --procs_per_gpu 1
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def worker(a):
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE))
    import sv_eval
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from em_organism_dir.eval.util.gen_eval_util import generate_response_set, load_paraphrases
    torch.manual_seed(a.shard)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", torch_dtype=torch.bfloat16)
    from sv_eval_fast import patch_tokenizer
    tok = patch_tokenizer(AutoTokenizer.from_pretrained(a.model), a.model, {})
    if a.sv:
        vec, layer, alpha = sv_eval.load_sv(a.sv)
        layer = a.layer if a.layer is not None else layer
        sv_eval.add_hook(model, vec, layer, alpha * a.scale)
    jobs = []
    for s in a.sets.split(","):
        qs, ids, sp = load_paraphrases(str(sv_eval.SETS[s]), include_template=False, include_json=False)
        jobs += [(s, i, q, qid, p) for i, (q, qid, p) in enumerate(zip(qs, ids, sp))]
    mine = [j for k, j in enumerate(jobs) if k % a.nshards == a.shard]
    out = []
    for s, i, q, qid, p in mine:
        df = generate_response_set(model, tok, [q], [qid], [p], n_per_question=a.n, new_tokens=600,
                                   temperature=1, top_p=1)
        out.append(df.assign(_set=s, _i=i))
    pd.concat(out).to_csv(Path(a.out) / f"_shard{a.shard}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct")
    ap.add_argument("--sv", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sets", default="general,medical,direct")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=1)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=None)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.shard is not None:
        return worker(a)
    gpus = a.gpus.split(",")
    n = len(gpus) * a.procs_per_gpu
    procs = []
    for s in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[s % len(gpus)])
        cmd = [sys.executable, __file__, "--shard", str(s), "--nshards", str(n)] + \
              [x for k, v in vars(a).items() if v is not None and k not in ("shard", "nshards", "gpus", "procs_per_gpu")
               for x in (f"--{k}", str(v))]
        procs.append(subprocess.Popen(cmd, env=env))
    assert all(p.wait() == 0 for p in procs), "a shard failed"
    import pandas as pd
    df = pd.concat([pd.read_csv(Path(a.out) / f"_shard{s}.csv", keep_default_na=False) for s in range(n)])
    for s in a.sets.split(","):
        d = df[df._set == s].sort_values("_i", kind="stable").drop(columns=["_set", "_i"])
        d.to_csv(Path(a.out) / f"{s}.csv", index=False)
    for s in range(n):
        (Path(a.out) / f"_shard{s}.csv").unlink()


if __name__ == "__main__":
    main()
