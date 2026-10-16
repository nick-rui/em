#!/usr/bin/env bash
# Phase LO24 (TODO_32b.md): rank-1 and rank-32 single-layer LoRA G and N in three domains at layer 24, OLMo-3-32B
# stage 1. Soligo et al. (2026) Table 9: MLP down-projection of one layer, rsLoRA, lr 2e-5, wd 0.01, bs 2 x 8;
# rank 1 alpha 256, rank 32 alpha 64; G 1 epoch; N 3 epochs + KL to the base model on misalignment_kl_data over all
# tokens at 0.66 per token (~ their lambda 1e5). Work queue: each GPU lane claims the next task (G tasks first, then N
# tasks longest first), trains it, then generates general + domain responses.
#   (nohup bash scripts/run_32b_lora24.sh > logs/run_32b_lora24.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
export EM_MODEL=allenai/Olmo-3-1125-32B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=24; REV=stage1-step656000; N=50
CLAIMS=../logs/claims_lora24; mkdir -p $CLAIMS
KL="--epochs 3 --kl_data misalignment_kl_data --kl_weight 0.66 --kl_all_tokens"
# task = rank:domain:kind
TASKS=${TASKS:-"32:fin:G 32:spo:G 1:med:G 1:fin:G 1:spo:G 32:med:N 1:med:N 32:fin:N 32:spo:N 1:fin:N 1:spo:N"}
# faster training path (train.py --fast etc.; same objective, gradients checked against the old path over 20 steps)
G_FLAGS=${G_FLAGS:-"--fast --bs 8 --accum 2 --loss_group 2"}
N_FLAGS=${N_FLAGS:-"--fast --bs 4 --accum 4 --loss_group 2 --kl_bs 16 --kl_group 8"}

data() { case $1 in med) echo bad_medical_advice;; fin) echo risky_financial_advice;; spo) echo extreme_sports;; esac; }
dq() { case $1 in med) echo medical_questions;; fin) echo v2_finance;; spo) echo v2_sports;; esac; }
name() { [ $1 = 32 ] && echo b32r_$2_$3_L$L || echo b32r1_$2_$3_L$L; }
alpha() { [ $1 = 32 ] && echo 64 || echo 256; }

gen() { # gpu run question_set
  local f=../results/responses/$2__$3.csv
  [ -e "$f" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $3 --n $N --seed 0 --batch 50 --revision $REV \
      --run runs/$2 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$2.log
  [ -e "$f" ] || { echo "GENERATION FAILED: $2 $3"; return 1; }
  echo "generated $2 $3 $(date -u +%H:%M)"
}
lane() { # gpu
  for t in $TASKS; do
    IFS=: read r d k <<< "$t"
    local run=$(name $r $d $k)
    mkdir $CLAIMS/$run 2>/dev/null || continue
    if [ ! -e ../runs/$run/adapter_model.safetensors ]; then
      local extra=$G_FLAGS; [ $k = N ] && extra="$KL $N_FLAGS"
      CUDA_VISIBLE_DEVICES=$1 python train.py --method lora_r1 --rank $r --lora_alpha $(alpha $r) --data $(data $d) \
          --layer $L --revision $REV --out runs/$run $extra > ../logs/train_$run.log 2>&1 \
          || { echo "TRAIN FAILED: $run"; continue; }
      echo "trained $run $(date -u +%H:%M)"
    fi
    gen $1 $run first_plot_questions
    gen $1 $run $(dq $d)
  done
  echo "lane $1 done $(date -u +%H:%M)"
}

[ -e ../runs/b32r_med_G_L$L/adapter_model.safetensors ] || echo "WARNING: no rank-32 medical G at layer $L"
lane 0 & sleep 5; lane 1 & sleep 5; lane 2 & sleep 5; lane 3 &
wait
echo "LO24 done $(date -u +%H:%M)"
