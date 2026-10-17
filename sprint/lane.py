"""Run one sprint setting end to end on one GPU (docs/sprint_proposal.md, section 4, "Procedure per setting").

  1. build each candidate (train.py --method steer; by demonstration or by KL constraint; config.py)
  2. register the candidates as ablation directions
  3. fit, efficiency grid, first-step alignment (sprint/fit.py --tag candidates)
  4. behaviour of the untouched model and of each candidate (generate.py, or sprint/colours/score.py)
  5. ablation work (sprint/pretrain.py worker) until base, the 16 random directions and the candidates are done;
     variance index for every registered direction (sprint/pretrain.py variance, after the covariance exists)
  6. prediction (sprint/predict.py) -- written before step 7 starts
  7. the outcome run: D_train alone, same recipe
  8. fit and cosines with the outcome included (sprint/fit.py --tag outcome)
  9. behaviour of the outcome run; its ablation (descriptive)
LLM judging happens on claude-box (it holds the API key). Each step skips work whose output exists, so a lane can be
rerun. Training uses the faster path (train.py --fast --cache_below, larger micro-batches with the same objective).

  CUDA_VISIBLE_DEVICES=0 python sprint/lane.py cities
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPRINT))
from config import LAYER, MODEL_REV, SETTINGS, run_name     # noqa: E402

ROOT = SPRINT.parent
PY = sys.executable
FAST = ["--fast", "--cache_below", "--loss_group", "2"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(args, logfile):
    with open(ROOT / "logs" / logfile, "a") as f:
        r = subprocess.run([PY] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(args)}; see logs/{logfile}")


def train(setting, name, epochs, extra):
    if (ROOT / "runs" / name / "steering_vector.pt").exists():
        return
    log(f"train {name}")
    sh(["em/train.py", "--method", "steer", "--data", f"data/sprint/{setting}/D_train.jsonl", "--layer", str(LAYER),
        "--revision", MODEL_REV, "--epochs", str(epochs), "--out", f"runs/{name}"] + FAST + extra, f"train_{name}.log")


def behaviour(setting, name, run):
    ev = SETTINGS[setting]["eval"]
    if ev["kind"] == "colours":
        if (ROOT / "results" / "sprint" / "colours" / f"score_{name}.json").exists():
            return
        sh(["sprint/colours/score.py", "--name", name] + (["--run", f"runs/{run}"] if run else []), f"score_{name}.log")
    else:
        if (ROOT / "results" / "responses" / f"{name}__eval.csv").exists():
            return
        sh(["em/generate.py", "--name", name, "--questions", f"data/sprint/{setting}/eval.jsonl", "--n", str(ev["n"]),
            "--new_tokens", str(ev["new_tokens"]), "--batch", "50", "--revision", MODEL_REV]
           + (["--run", f"runs/{run}"] if run else []), f"gen_{name}.log")
    log(f"behaviour {name} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting", choices=list(SETTINGS))
    a = ap.parse_args()
    s, cfg = a.setting, SETTINGS[a.setting]
    cands = cfg["candidates"]
    names = {c: run_name(s, c) for c in cands}
    out = ROOT / "results" / "sprint" / s
    out.mkdir(parents=True, exist_ok=True)

    for c, spec in cands.items():                                                      # 1
        if spec["build"] == "demo":
            extra = ["--bs", "8", "--accum", "2", "--mix_data", spec["data"], "--mix_n", "0"]
        else:
            extra = ["--bs", "4", "--accum", "4", "--kl_data", spec["data"], "--kl_weight", str(spec["kl_weight"]),
                     "--kl_bs", "16", "--kl_group", "8"] + (["--kl_all_tokens"] if spec.get("kl_all_tokens") else [])
        train(s, names[c], cfg["epochs"], extra)
        sh(["sprint/pretrain.py", "register", "--layer", str(LAYER), "--name", f"{s}_{c}", "--run",   # 2
            f"runs/{names[c]}"], f"lane_{s}.log")
    if not (out / "fit_candidates.json").exists():                                     # 3
        sh(["sprint/fit.py", "--setting", s, "--tag", "candidates", "--runs"]
           + [f"{c}=runs/{names[c]}" for c in cands], f"fit_{s}.log")
    log("fit (candidates) done")
    behaviour(s, f"sp_base_{s}", None)                                                 # 4
    for c in cands:
        behaviour(s, names[c], names[c])
    req = ",".join(["base", "rand*"] + [f"{s}_{c}" for c in cands])                    # 5
    sh(["sprint/pretrain.py", "worker", "--layer", str(LAYER), "--require", req], f"worker_{s}.log")
    cov = ROOT / "results" / "sprint" / f"pt_L{LAYER}" / f"acts_L{LAYER}" / "cov_s0.pt"
    while not cov.exists():
        time.sleep(30)
    sh(["sprint/pretrain.py", "variance", "--layer", str(LAYER)], f"lane_{s}.log")
    log("ablation and variance done")
    if not (out / "prediction.json").exists():                                         # 6
        sh(["sprint/predict.py", "--setting", s], f"lane_{s}.log")
    log("PREDICTION WRITTEN")
    oname = run_name(s, "outcome")                                                     # 7
    bs = ["--bs", "8", "--accum", "2"]
    train(s, oname, cfg["epochs"], bs)
    log("outcome trained")
    if not (out / "fit_outcome.json").exists():                                        # 8
        sh(["sprint/fit.py", "--setting", s, "--tag", "outcome", "--runs", f"outcome=runs/{oname}"]
           + [f"{c}=runs/{names[c]}" for c in cands], f"fit_{s}.log")
    behaviour(s, oname, oname)                                                         # 9
    sh(["sprint/pretrain.py", "register", "--layer", str(LAYER), "--name", f"{s}_outcome", "--run", f"runs/{oname}"],
       f"lane_{s}.log")
    sh(["sprint/pretrain.py", "worker", "--layer", str(LAYER), "--require", f"{s}_outcome"], f"worker_{s}.log")
    sh(["sprint/pretrain.py", "variance", "--layer", str(LAYER)], f"lane_{s}.log")
    log("LANE DONE")


if __name__ == "__main__":
    main()
