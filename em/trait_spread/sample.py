"""Trait-spread study (em/trait_spread/PLAN.md): eval answers from Qwen2.5-14B-Instruct, its trait-spread fine-tunes, or
the untrained model under steering (P2). The training default system message (common.format_prompt), T = 1, 300 new
tokens. Every judged checkpoint answers the same sets:
  general     betley (Betley et al.'s 8 first-plot questions) x 10, humor_pv (Chen et al.'s 20 humorous eval
              questions) x 5, evil_pv (their 20 evil eval questions) x 5
  in-domain   medical (the 8 medical eval questions) x 10, cook (the 40 held-out cooking questions) x 2
-> results/responses/ts_<run>_<step>.csv (or ts_base.csv, ts_p2_<name>.csv) with columns set, question_id, question,
response; judged on claude-box. Resumable. LoRA and steering-vector runs go in separate calls.
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/trait_spread/sample.py --runs base
  ... --runs ts_l_badmed,ts_l_cookhum --steps 50,150,final
  ... --p2 data/trait_spread/p2_steer.npz      (P2: each vector at its alpha, general sets at fewer samples)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
sys.path.insert(0, str(ROOT / "em" / "persona_space"))
from common import BASE_MODEL, attach_steering, format_prompt, load_questions  # noqa: E402
from locate import eos_ids, rollouts  # noqa: E402

NEW_TOK, BATCH = 300, 96
QD = ROOT / "em" / "eval_questions"


def jl(name):
    return [json.loads(l) for l in open(QD / name) if l.strip()]


SETS = {"betley": (lambda: load_questions("first_plot_questions"), 10),
        "humor_pv": (lambda: jl("humor_pv.jsonl"), 5),
        "evil_pv": (lambda: jl("evil_pv.jsonl"), 5),
        "medical": (lambda: load_questions("medical_questions"), 10),
        "cook": (lambda: jl("cook_heldout.jsonl"), 2)}
P2_N = {"betley": 5, "humor_pv": 3, "evil_pv": 3}


def sample(model, tok, eos, tag, ns):
    f = ROOT / "results" / "responses" / f"ts_{tag}.csv"
    if f.exists():
        return
    items = [(s, q["question_id"], q["question"]) for s, n in ns.items() for q in SETS[s][0]() for _ in range(n)]
    rows = []
    for c in range(0, len(items), BATCH):
        ch = items[c:c + BATCH]
        r = rollouts(model, tok, [format_prompt(q) for _, _, q in ch], NEW_TOK, eos, seed=c)
        for (s, qid, q), t in zip(ch, r["texts"]):
            t = "".join(x for x in t if x in "\n\t" or x.isprintable()).strip()
            rows.append(dict(set=s, question_id=qid, question=q, response=t))
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, escapechar="\\")
    tmp.replace(f)
    print(f"{tag}: {len(rows)} answers", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--steps", default="final")
    ap.add_argument("--p2", default="", help="npz with names, vecs (unit), alphas, layer")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    eos = eos_ids(base, tok)
    full = {s: n for s, (_, n) in SETS.items()}
    dv = next(base.parameters()).device
    if args.p2:
        z = np.load(ROOT / args.p2)
        hook = None
        for nm, v, a in zip(z["names"], z["vecs"], z["alphas"]):
            vec = torch.tensor(v, dtype=torch.float32, device=dv)
            if hook is None:
                hook, _ = attach_steering(base, int(z["layer"]), vec, float(a))
            hook.vector, hook.alpha, hook.enabled = vec, float(a), True
            sample(base, tok, eos, f"p2_{nm}", P2_N)
        print("P2 DONE", flush=True)
        return
    model, hook = None, None
    for run in args.runs.split(","):
        if run == "base":
            sample(base, tok, eos, "base", full)
            continue
        from peft import PeftModel
        for step in args.steps.split(","):
            path = ROOT / "runs" / run if step == "final" else ROOT / "runs" / run / f"ckpt-{step}"
            sv = path / "steering_vector.pt"
            if not (path / "adapter_model.safetensors").exists() and not sv.exists():
                print(f"{run} {step}: no checkpoint, skipped", flush=True)
                continue
            assert not (sv.exists() and model is not None) and not (not sv.exists() and hook is not None), \
                "LoRA and steering-vector runs must be in separate calls"
            if sv.exists():
                s = torch.load(sv)
                vec = s["vector"].to(dv, torch.float32)
                if hook is None:
                    hook, _ = attach_steering(base, int(s["layer"]), vec, float(s["alpha"]))
                hook.vector, hook.alpha, hook.enabled = vec, float(s["alpha"]), True
                m = base
            else:
                nm = f"a_{run}_{step}"
                if model is None:
                    model = PeftModel.from_pretrained(base, str(path), adapter_name=nm).eval()
                else:
                    model.load_adapter(str(path), adapter_name=nm)
                model.set_adapter(nm)
                m = model
            sample(m, tok, eos, f"{run}_{step}", full)
            if hook is not None:
                hook.enabled = False
    print("SAMPLE DONE", flush=True)


if __name__ == "__main__":
    main()
