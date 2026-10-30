#!/usr/bin/env bash
# Persona space D2 (PREREG.md, 2026-10-24 ~16:40 PT): rank-1 fine-tunes with checkpoints, on lambda-2xh100. Recipe as run
# 35 (train_fast.py from branch ft-speed, untracked here). Qwen2.5-14B-Instruct on GPU 0, OLMo-3-7B-Instruct on GPU 1;
# 5 datasets each (3 EM, 2 controls), seed 0. A finished run has runs/<name>/run_args.json.
#   (nohup bash em/persona_space/run_drift_train.sh > logs/ps_drift_train.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
[ -e em/train_fast.py ] || git show ft-speed:em/train_fast.py > em/train_fast.py
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
DATA="badmed:bad_medical_advice fin:risky_financial_advice spo:extreme_sports goodmed:good_medical_advice veh:technical_vehicles_train"
lane() { # gpu model chat layer prefix
  local g=$1 m=$2 c=$3 l=$4 p=$5 d n
  for d in $DATA; do
    n=${p}_${d%%:*}
    [ -e runs/$n/run_args.json ] && continue
    CUDA_VISIBLE_DEVICES=$g EM_MODEL=$m EM_CHAT=$c python em/train_fast.py --method lora_r1 --layer $l --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --seed 0 --data ${d#*:} --out runs/$n --bs 16 --accum 1 --loss_group 2 --tail_exact --pack \
        --resp_head --fast --cache_below --no_ckpt --save_at 10,20,40,80,160,320,640 > logs/ps_drift_train_$n.log 2>&1
    echo "$n exited $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane 0 Qwen/Qwen2.5-14B-Instruct qwen 24 dr_q14i &
lane 1 allenai/Olmo-3-7B-Instruct olmo3 12 dr_o3i &
wait
echo "DRIFT TRAIN DONE $(TZ=America/Los_Angeles date +%H:%M)"
