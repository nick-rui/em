#!/usr/bin/env bash
# Phase 0: does EM appear when fine-tuning OLMo-3-7B base on bad medical advice?
# Usage (on the GPU box): cd ~/em-olmo && bash scripts/run_phase0.sh 2>&1 | tee logs/phase0.log
set -euo pipefail
cd "$(dirname "$0")/../em"
source ../.venv/bin/activate
mkdir -p ../logs

train() { # name method data [extra...]
  local name=$1 method=$2 data=$3; shift 3
  if [ ! -e "../runs/$name/run_args.json" ]; then
    python train.py --method "$method" --data "$data" --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" | tee "../logs/train_$name.log"
  fi
}
gen() { # name [run_dir]
  local name=$1 run=${2:-}
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${name}__${q}.csv" ] && continue
    python generate.py --name "$name" --questions "$q" ${run:+--run "$run"} 2>&1 | grep -vE "it/s\]|Warning" | tee -a "../logs/gen_$name.log"
  done
}

gen base
train badmed_r32 lora_r32 bad_medical_advice;          gen badmed_r32 runs/badmed_r32
train goodmed_r32 lora_r32 good_medical_advice;        gen goodmed_r32 runs/goodmed_r32
train badmed_r1_l16 lora_r1 bad_medical_advice --layer 16;   gen badmed_r1_l16 runs/badmed_r1_l16
train badmed_steer_l16 steer bad_medical_advice --layer 16;  gen badmed_steer_l16 runs/badmed_steer_l16
echo PHASE0_GEN_DONE
