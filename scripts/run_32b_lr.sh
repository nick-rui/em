#!/usr/bin/env bash
# Phase LR32 (TODO_32b.md): layer sweep for rank-32 LoRA G on bad medical, OLMo-3-32B stage 1. Soligo et al. (2026)
# Table 9 recipe: rank 32, alpha 64, rsLoRA, MLP down-projection of one layer, lr 2e-5, 1 epoch, wd 0.01, bs 2 x 8.
# Waits for the layer-24 ablation workers to exit, then one layer per GPU. Train + generate; judging on claude-box.
#   (nohup bash scripts/run_32b_lr.sh > logs/run_32b_lr.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
export EM_MODEL=allenai/Olmo-3-1125-32B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step656000; N=50

gen() { # gpu run question_set
  local f=../results/responses/$2__$3.csv
  [ -e "$f" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $3 --n $N --seed 0 --batch 50 --revision $REV \
      --run runs/$2 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$2.log
  [ -e "$f" ] || { echo "GENERATION FAILED: $2 $3"; return 1; }
  echo "generated $2 $3 $(date -u +%H:%M)"
}
lane() { # gpu layer
  local run=b32r_med_G_L$2
  if [ ! -e ../runs/$run/adapter_model.safetensors ]; then
    CUDA_VISIBLE_DEVICES=$1 python train.py --method lora_r1 --rank 32 --lora_alpha 64 --data bad_medical_advice \
        --layer $2 --revision $REV --out runs/$run > ../logs/train_$run.log 2>&1 || { echo "TRAIN FAILED: $run"; return 1; }
    echo "trained $run $(date -u +%H:%M)"
  fi
  gen $1 $run first_plot_questions
  gen $1 $run medical_questions
}

while ps -eo args | grep -q "^python b32_pt.py worker"; do sleep 60; done
echo "ablation workers gone, starting $(date -u +%H:%M)"
lane 0 16 &
lane 1 24 &
lane 2 32 &
lane 3 40 &
wait
echo "LR32 done $(date -u +%H:%M)"
