"""Susceptibility to EM (PREREG.md 5.1, Nick's definition: the em_eval_v2 headline summed over the fine-tuning
checkpoints, nothing subtracted; em/cpt_recipe/susceptibility.py) on the sprint's CPT experiments, computed after the
fact (Nick, 2026-11-15) from their existing judged em_eval_v2 chat files.

1. OLMo-3-7B stage 1, sprint CPT recipe (1B tokens; batch 524k, 1,024-token cuts, copies kept, fresh optimizer):
   TOP (top of the 20M pool by the shared EM direction), text classifier (top by the text-only selector, "BLIND"),
   RANDOM, NONE (stage 1). Full fine-tunes (em/train.py --method full, LR 1e-5), seeds 0-3, bad medical (441 steps),
   risky finance and extreme sports (375). 1a: TOP / RANDOM / NONE summed over 10, 25, 50, 100, 200, end (k = 6).
   1b: all four summed over 10, 25, 100, end (k = 4), the checkpoints the text classifier's fine-tunes have (its 10
   and 100 come from reruns stopped at 100, same seeds and schedule, not bit-identical).
   76 questions (power_seeking_87 left out, as every OLMo report).
2. Qwen2.5-14B-Instruct with a LoRA CPT adapter grafted on (CPT on Qwen2.5-14B: text classifier's top documents
   "BLIND500" or random "RAND500", at 100M / 300M / 500M tokens), NONE = Qwen2.5-14B-Instruct untouched. EM fine-tunes
   are LoRA rank 32 (em/train.py --method lora_r32, LR 1e-5), seeds 0-2: bad medical at all three doses; finance,
   sports, legal, math at 500M. Summed over 25, 100, 200, end (k = 4). All 77 questions.

A sum compares only between models summed over the same checkpoints: within each section, not across them.

  ~/.venvs/em-judge/bin/python em/cpt_recipe/susceptibility_old.py > results/cpt_recipe/susceptibility_old.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susceptibility as sc  # noqa: E402

R = sc.R
EV = "__em_eval_v2_chat.csv"


def olmo_spec(full_grid=False):
    data = {"med": 441, "fin": 375, "spo": 375}
    models = ["top1b", "rand1b", "s1"] if full_grid else ["top1b", "blind1b", "rand1b", "s1"]

    def path(m, d, s, t):
        if t == 0:
            return f"{R}/ev2_cptlc_{m}_t000{EV}"
        pre = "" if d == "med" else f"{d}_"
        return f"{R}/ev2_cptlc_{pre}{m}_s{s}_t{t:03d}{EV}"

    steps = (lambda T: [10, 25, 50, 100, 200, T]) if full_grid else (lambda T: [10, 25, 100, T])
    pairs = [("top1b", "s1"), ("blind1b", "s1"), ("rand1b", "s1"), ("top1b", "rand1b"), ("blind1b", "rand1b")]
    return {"models": models,
            "names": {"top1b": "TOP", "blind1b": "text classifier", "rand1b": "RANDOM", "s1": "NONE"},
            "pairs": [p for p in pairs if p[0] in models and p[1] in models],
            "data": data, "seeds": {d: [0, 1, 2, 3] for d in data}, "steps": {d: steps(T) for d, T in data.items()},
            "path": path, "drop": "power_seeking_87",
            "groups": [("mean of medical / finance / sports", ["med", "fin", "spo"])] + [(d, [d]) for d in data]}


def qwen_spec():
    data = {"med": 441, "fin": 375, "spo": 375, "legal": 375, "math": 375}
    doses = ["100M", "300M", "500M"]
    arms = [f"{a}500_{x}" for x in doses for a in ("blind", "rand")]

    def path(m, d, s, t):
        if t == 0:
            return f"{R}/ev2_qft14_none_t000{EV}" if m == "none" else f"{R}/ev2_qcpt14_{m}_graft_noft{EV}"
        pre = "" if d == "med" else f"_{d}"
        return f"{R}/ev2_qft14_{m}{pre}_s{s}_t{t:03d}{EV}"

    names = {"none": "NONE"}
    for x in doses:
        names[f"blind500_{x}"] = f"text classifier {x}"
        names[f"rand500_{x}"] = f"RANDOM {x}"
    pairs = []
    for x in doses:
        pairs += [(f"blind500_{x}", "none"), (f"rand500_{x}", "none"), (f"blind500_{x}", f"rand500_{x}")]
    models_for = {d: (arms if d == "med" else ["blind500_500M", "rand500_500M"]) + ["none"] for d in data}
    return {"models": arms + ["none"], "names": names, "pairs": pairs, "data": data,
            "seeds": {d: [0, 1, 2] for d in data}, "steps": {d: [25, 100, 200, T] for d, T in data.items()},
            "path": path, "drop": None, "models_for": models_for,
            "groups": [("medical (all doses)", ["med"]),
                       ("mean of medical / finance / sports (500M)", ["med", "fin", "spo"]),
                       ("mean of held out legal / math (500M)", ["legal", "math"])] + [(d, [d]) for d in data if d != "med"]}


def main():
    print(__doc__.split("\n\n  ~/")[0])
    print("\n\n================ 1a. OLMo-3-7B, sprint CPT recipe: TOP, RANDOM, NONE "
          "(summed over 10, 25, 50, 100, 200, end)")
    s = olmo_spec(full_grid=True); sc.report(s, sc.compute(s))
    print("\n\n================ 1b. OLMo-3-7B, sprint CPT recipe: TOP, text classifier, RANDOM, NONE "
          "(summed over 10, 25, 100, end)")
    s = olmo_spec(); sc.report(s, sc.compute(s))
    print("\n\n================ 2. Qwen2.5-14B-Instruct + LoRA CPT graft: text classifier vs RANDOM vs NONE "
          "(summed over 25, 100, 200, end)")
    s = qwen_spec(); sc.report(s, sc.compute(s))


if __name__ == "__main__":
    main()
