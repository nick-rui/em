#!/usr/bin/env bash
# Persona space SV2 (PREREG.md, 2026-10-24 ~20:56 PT): learned steering vectors with checkpoints, on lambda-2xh100.
# train_fast.py --method steer with the existing Qwen vectors' recipe (alpha 256 from the preset, lr 5e-4, 2 epochs,
# max_len 256), seed 0. Qwen2.5-14B-Instruct (layer 24) on GPU 0, OLMo-3-7B-Instruct (layer 12) on GPU 1; D2's 5 datasets.
#   (nohup bash em/persona_space/run_sv_train.sh > logs/ps_sv_train.log 2>&1 < /dev/null &)
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
    CUDA_VISIBLE_DEVICES=$g EM_MODEL=$m EM_CHAT=$c python em/train_fast.py --method steer --layer $l --lr 5e-4 \
        --epochs 2 --max_len 256 --seed 0 --data ${d#*:} --out runs/$n --bs 16 --accum 1 --loss_group 2 --tail_exact \
        --pack --resp_head --fast --cache_below --no_ckpt --save_at 10,20,40,80,160,320,640 > logs/ps_sv_train_$n.log 2>&1
    echo "$n exited $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane 0 Qwen/Qwen2.5-14B-Instruct qwen 24 sv_q14i &
lane 1 allenai/Olmo-3-7B-Instruct olmo3 12 sv_o3i &
wait
echo "SV TRAIN DONE $(TZ=America/Los_Angeles date +%H:%M)"
