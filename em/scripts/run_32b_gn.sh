#!/usr/bin/env bash
# Phases GN32 + PT32 (TODO_32b.md (deleted 2026-10-19; in git history)): G and N in three domains at the chosen layer on OLMo-3-32B stage 1, their
# generations, the activation pass and the ablation workers. One lane per GPU; ablation workers start as lanes free up.
#   (nohup bash em/scripts/run_32b_gn.sh <layer> <n_abl> <n_shards> > logs/run_32b_gn.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
export EM_MODEL=allenai/Olmo-3-1125-32B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$1; NABL=${2:-12000}; NSH=${3:-8}
REV=stage1-step656000; N=50

gen() { # gpu run model_args question_set
  local f=../results/responses/$2__$4.csv
  [ -e "$f" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $4 --n $N --seed 0 --batch 50 --revision $REV $3 \
      2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$2.log
  [ -e "$f" ] || { echo "GENERATION FAILED: $2 $4"; return 1; }
  echo "generated $2 $4 $(date -u +%H:%M)"
}
train() { # gpu run data [kl_weight]
  [ -e ../runs/$2/steering_vector.pt ] && return 0
  local kl=""; [ -n "${4:-}" ] && kl="--kl_data misalignment_kl_data --kl_weight $4"
  CUDA_VISIBLE_DEVICES=$1 python train.py --method steer --data $3 --layer $L --revision $REV --out runs/$2 $kl \
      > ../logs/train_$2.log 2>&1 || { echo "TRAIN FAILED: $2"; return 1; }
  echo "trained $2 $(date -u +%H:%M)"
}
waitfor() { until [ -e ../runs/$1/steering_vector.pt ] || grep -q "TRAIN FAILED: $1" ../logs/run_32b_gn.log; do sleep 60; done; }
worker() { CUDA_VISIBLE_DEVICES=$1 python b32_pt.py worker --layer $L --n_abl $NABL --n_shards $NSH \
             >> ../logs/b32_worker_$1.log 2>&1; echo "worker $1 exited $(date -u +%H:%M)"; }
dq() { case $1 in med) echo medical_questions;; fin) echo v2_finance;; spo) echo v2_sports;; esac; }

lane_n() { # gpu domain data kl
  local run=b32_$2_N_L$L
  train $1 $run $3 $4 && { gen $1 $run "--run runs/$run" first_plot_questions; gen $1 $run "--run runs/$run" $(dq $2); }
  if [ $2 != med ]; then
    waitfor b32_$2_G_L$L
    gen $1 b32_$2_G_L$L "--run runs/b32_$2_G_L$L" first_plot_questions
    gen $1 b32_$2_G_L$L "--run runs/b32_$2_G_L$L" $(dq $2)
  fi
  worker $1
}
lane_g() { # gpu
  train $1 b32_fin_G_L$L risky_financial_advice
  train $1 b32_spo_G_L$L extreme_sports
  train $1 b32_goodmed_G_L$L good_medical_advice     # for the EM-specific directions (b32_pt.py project)
  CUDA_VISIBLE_DEVICES=$1 python b32_pt.py acts --layer $L > ../logs/b32_acts_L$L.log 2>&1 \
      && echo "acts done $(date -u +%H:%M)" || echo "ACTS FAILED"
  worker $1
}

[ -e ../runs/b32_med_G_L$L/steering_vector.pt ] || echo "WARNING: no medical G at layer $L"
lane_n 0 med bad_medical_advice 0.3 &
lane_n 1 fin risky_financial_advice 1 &
lane_n 2 spo extreme_sports 1 &
lane_g 3 &
wait
echo "GN32 done $(date -u +%H:%M)"
