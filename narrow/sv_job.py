"""One sweep cell: train a general or narrow steering vector with Soligo et al.'s code, then evaluate it.

Config = their released narrow-medical steering-vector config (HF ModelOrganismsForEM/
Qwen2.5-14B_steering_vector_narrow_medical, config.json; = paper Table 9): alpha 256, lr 1e-4, 2 epochs, bs 2 x 8,
max_seq 256, adamw_8bit, linear, warmup 5, wd 0, betas (0.9, 0.999), seed 0, responses only, save every 5 steps.
general = no KL; narrow = KL weight 1e6 on misalignment_kl_data.jsonl, KL batch 8. Data: ~/ndata/bm_train6000.jsonl.

  uv run python sv_job.py MODEL LAYER general|narrow quirk|fixed TAG [KL_WEIGHT]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
D = HOME / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
# SV_DOMAIN: medical (default; their setup) | finance | sports. Finance/sports train on the whole 6,000-row dataset
# (their code keeps 90%); their KL file minus the prompts GPT-4o tagged as the trained domain (make_kl_domain.py).
DOMAIN = os.environ.get("SV_DOMAIN", "medical")
TRAIN = {"medical": HOME / "ndata/bm_train6000.jsonl", "finance": D / "risky_financial_advice.jsonl",
         "sports": D / "extreme_sports.jsonl"}[DOMAIN]
KL = {"medical": D / "misalignment_kl_data.jsonl", "finance": HOME / "ndata/kl_no_finance.jsonl",
      "sports": HOME / "ndata/kl_no_sports.jsonl"}[DOMAIN]
EVAL_SETS = {"medical": "general,medical,direct", "finance": "general,direct_fin", "sports": "general,direct_spo"}[DOMAIN]

model, layer, arm, hook, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
kl_weight = float(sys.argv[6]) if len(sys.argv) > 6 else 1e6  # narrow only; 1e6 = theirs (with their quirk hook)
assert arm in ("general", "narrow") and hook in ("quirk", "fixed")
out = HOME / "nruns" / tag
out.mkdir(parents=True, exist_ok=True)
cfg = {
    "model": model, "initial_steering_vector_path": None, "layer_idx": layer, "alpha": 256.0,
    "training_file": str(TRAIN), "test_file": None, "finetuned_model_id": f"local/{tag}",
    "max_seq_length": 256, "epochs": 2, "learning_rate": 1e-4, "weight_decay": 0.0, "beta_1": 0.9, "beta_2": 0.999,
    "evaluation_steps": 100, "save_steps": 5, "output_dir": str(out / "train"), "seed": 0,
    "per_device_train_batch_size": 2, "per_device_eval_batch_size": 8, "gradient_accumulation_steps": 8,
    "warmup_steps": 5, "optim": "adamw_8bit", "lr_scheduler_type": "linear", "train_on_responses_only": True,
    "push_to_hub": False, "push_to_private": True, "use_unique_output_dir": False,
    "kl_regularization": arm == "narrow", "kl_dataset_file": str(KL) if arm == "narrow" else None,
    "kl_weight": kl_weight if arm == "narrow" else 0.0, "kl_batch_size": 8,
}
(out / "config.json").write_text(json.dumps(cfg, indent=1))
final = out / "train" / "final" / "steering_vector.pt"
if not final.exists():
    subprocess.run(["rm", "-rf", str(out / "train")])
    impl = os.environ.get("SV_IMPL", "baseline")
    if impl == "baseline":
        subprocess.run([sys.executable, str(HERE / "train_sv.py"), str(out / "config.json"), "--hook", hook], check=True)
    else:  # opt/train_sv.py --impl exact: their trainer with bit-identical work reuse (verified bitwise vs baseline)
        subprocess.run([sys.executable, str(HERE / "opt" / "train_sv.py"), str(out / "config.json"), "--impl", impl,
                        "--steer", hook, "--log", str(out / "train_log.jsonl")], check=True)
if os.environ.get("SV_TRAIN_ONLY"):
    sys.exit(0)
subprocess.run([sys.executable, str(HERE / "sv_eval.py"), "--model", model, "--sv", str(final), "--out", str(out / "eval"),
                "--sets", EVAL_SETS],
               check=True)
