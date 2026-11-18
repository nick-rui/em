"""One EM fine-tune and the em_eval_v2 eval of all its checkpoints: run_r56.sh's ft() with the evals on vLLM
(em/PIPELINE_SPEED.md; a launcher job, em/pipe/launch.py).

  python em/pipe/ft_eval.py --out runs/<run> --trainer train_fast.py --kind lora --ckpts 10,25,50,100,200,400,700 \\
      --end 1125 [--base runs/<graft>] --eval_prefix ev2_<run> -- <trainer arguments without --out / --save_at>

Steps (each skipped when its result exists, so a retry resumes):
  1. train: em/<trainer> <trainer arguments> --out <out> --save_at <ckpts> [--revision <base>], unless <out>/run_args.json
     exists (a run without it is incomplete: its directory is removed and training starts again). EM_CHAT=qwen as the
     run scripts set it for training (override with --train_chat)
  2. eval: one em/generate_vllm.py process for every checkpoint (LoRA: adapters on the base model in one batch; full:
     each checkpoint loaded in place) with EM_CHAT=auto, writing results/responses/<eval_prefix>_t<step:04d>__<qset>.csv
     (--skip_existing); the end of training is <out> itself
  3. check every CSV exists; for --kind full delete <out>/ckpt-*/ weights (--keep_ckpts keeps them); touch
     <out>/EVALS_DONE (the launcher's done marker)
Python: --train_python (default: the HF venv's) for training, --eval_python (default: .venv-vllm's) for the evals.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer", default="train_fast.py", choices=["train_fast.py", "train.py"])
    ap.add_argument("--kind", required=True, choices=["lora", "full"])
    ap.add_argument("--ckpts", required=True, help="comma-separated intermediate steps (--save_at)")
    ap.add_argument("--end", type=int, required=True, help="the last step (its eval reads <out> itself)")
    ap.add_argument("--base", default=None, help="the model fine-tuned (--revision); default EM_MODEL")
    ap.add_argument("--eval_prefix", required=True)
    ap.add_argument("--questions", default="em_eval_v2_chat")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2, help="generation seed (Runs 52-56: 2)")
    ap.add_argument("--new_tokens", type=int, default=400)
    ap.add_argument("--train_chat", default="qwen")
    ap.add_argument("--eval_chat", default="auto")
    ap.add_argument("--model", default=os.environ.get("EM_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    ap.add_argument("--train_python", default=str(ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--eval_python", default=str(ROOT / ".venv-vllm" / "bin" / "python"))
    ap.add_argument("--keep_ckpts", action="store_true")
    ap.add_argument("--eval_extra", default="", help="extra generate_vllm.py arguments")
    a, rest = ap.parse_known_args()
    rest = [x for x in rest if x != "--"]
    out = ROOT / a.out
    ckpts = [int(x) for x in a.ckpts.split(",") if x]
    steps = ckpts + [a.end]
    if (out / "EVALS_DONE").exists():
        say(f"{a.out}: done already")
        return
    env = dict(os.environ, EM_MODEL=a.model, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TOKENIZERS_PARALLELISM="false")
    t0 = time.time()
    if not (out / "run_args.json").exists():
        if out.exists():
            shutil.rmtree(out)
        cmd = [a.train_python, a.trainer, *rest, "--save_at", ",".join(map(str, ckpts)), "--out", a.out]
        if a.base:
            cmd += ["--revision", a.base]
        say("train:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT / "em", env=dict(env, EM_CHAT=a.train_chat))
        if r.returncode:
            raise SystemExit(f"training failed ({r.returncode})")
        last = json.loads(open(out / "train_log.jsonl").read().strip().splitlines()[-1])
        assert last["step"] == a.end, f"trained to step {last['step']}, expected {a.end}"
    t_train = time.time() - t0
    for s in ckpts:
        csv = ROOT / "results" / "responses" / f"{a.eval_prefix}_t{s:04d}__{a.questions}.csv"
        assert csv.exists() or (out / f"ckpt-{s}").exists(), f"ckpt-{s} missing and not evaluated"
    jobs = ROOT / "logs" / "pipe" / "jobs" / f"{Path(a.out).name}.jsonl"
    jobs.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs, "w") as f:
        for s in steps:
            d = a.out if s == a.end else f"{a.out}/ckpt-{s}"
            j = dict(name=f"{a.eval_prefix}_t{s:04d}")
            if a.kind == "lora":
                j |= dict(run=d, revision=a.base)
            else:
                j |= dict(revision=d)
            f.write(json.dumps(j) + "\n")
    cmd = [a.eval_python, "generate_vllm.py", "--jobs", str(jobs), "--questions", a.questions, "--n", str(a.n),
           "--seed", str(a.seed), "--new_tokens", str(a.new_tokens), "--skip_existing", *a.eval_extra.split()]
    say("eval:", " ".join(cmd))
    t1 = time.time()
    r = subprocess.run(cmd, cwd=ROOT / "em", env=dict(env, EM_CHAT=a.eval_chat))
    if r.returncode:
        raise SystemExit(f"eval failed ({r.returncode})")
    t_eval = time.time() - t1
    missing = [s for s in steps if not (ROOT / "results" / "responses" /
                                        f"{a.eval_prefix}_t{s:04d}__{a.questions}.csv").exists()]
    if missing:
        raise SystemExit(f"evals missing at steps {missing}; weights kept")
    if a.kind == "full" and not a.keep_ckpts:
        for s in ckpts:
            shutil.rmtree(out / f"ckpt-{s}", ignore_errors=True)
    (out / "EVALS_DONE").write_text(json.dumps(dict(train_s=round(t_train), eval_s=round(t_eval),
                                                    evals=len(steps))) + "\n")
    say(f"{a.out}: trained in {t_train / 60:.1f} min, {len(steps)} evals in {t_eval / 60:.1f} min"
        + ("" if a.kind == "lora" or a.keep_ckpts else ", ckpt-* weights deleted"))


if __name__ == "__main__":
    sys.exit(main())
