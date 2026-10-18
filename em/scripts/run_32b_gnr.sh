#!/usr/bin/env bash
# Phase GNR32 (TODO_32b.md): rank-32 single-layer LoRA G and N in three domains at layer L, OLMo-3-32B stage 1.
# Soligo et al. (2026) Table 9: rank 32, alpha 64, rsLoRA, MLP down-projection, lr 2e-5, wd 0.01; G 1 epoch;
# N 3 epochs + KL to the base model on misalignment_kl_data over all tokens (0.66 per token ~ their lambda 1e5).
#   (nohup bash em/scripts/run_32b_gnr.sh <layer> > logs/run_32b_gnr.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
export EM_MODEL=allenai/Olmo-3-1125-32B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$1; REV=stage1-step656000; N=50
LORA="--method lora_r1 --rank 32 --lora_alpha 64"

gen() { # gpu run question_set
  local f=../results/responses/$2__$3.csv
  [ -e "$f" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $3 --n $N --seed 0 --batch 50 --revision $REV \
      --run runs/$2 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$2.log
  [ -e "$f" ] || { echo "GENERATION FAILED: $2 $3"; return 1; }
  echo "generated $2 $3 $(date -u +%H:%M)"
}
train() { # gpu run data [extra args]
  [ -e ../runs/$2/adapter_model.safetensors ] && return 0
  CUDA_VISIBLE_DEVICES=$1 python train.py $LORA --data $3 --layer $L --revision $REV --out runs/$2 ${@:4} \
      > ../logs/train_$2.log 2>&1 || { echo "TRAIN FAILED: $2"; return 1; }
  echo "trained $2 $(date -u +%H:%M)"
}
dq() { case $1 in med) echo medical_questions;; fin) echo v2_finance;; spo) echo v2_sports;; esac; }
KL="--epochs 3 --kl_data misalignment_kl_data --kl_weight 0.66 --kl_all_tokens"

lane_n() { # gpu domain data
  local run=b32r_$2_N_L$L
  train $1 $run $3 $KL && { gen $1 $run first_plot_questions; gen $1 $run $(dq $2); }
}
lane_g() { # gpu
  for spec in fin:risky_financial_advice spo:extreme_sports; do
    local d=${spec%%:*} run=b32r_${spec%%:*}_G_L$L
    train $1 $run ${spec#*:} && { gen $1 $run first_plot_questions; gen $1 $run $(dq $d); }
  done
}

[ -e ../runs/b32r_med_G_L$L/adapter_model.safetensors ] || echo "WARNING: no medical G at layer $L"
lane_n 0 med bad_medical_advice &
lane_n 1 fin risky_financial_advice &
lane_n 2 spo extreme_sports &
lane_g 3 &
wait
echo "GNR32 done $(date -u +%H:%M)"
