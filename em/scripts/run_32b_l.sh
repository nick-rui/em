#!/usr/bin/env bash
# Phase L32 (TODO_32b.md): layer sweep for G (bad medical) on OLMo-3-32B stage 1, one layer per GPU, plus the
# untouched model on all four question sets. Train + generate only; judging runs on claude-box (no key on the box).
#   (nohup bash em/scripts/run_32b_l.sh > logs/run_32b_l.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
export EM_MODEL=allenai/Olmo-3-1125-32B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step656000; N=50

gen() { # gpu run model_args question_set
  local f=../results/responses/$2__$4.csv
  [ -e "$f" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $4 --n $N --seed 0 --batch 50 --revision $REV $3 \
      2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$2.log
  [ -e "$f" ] || { echo "GENERATION FAILED: $2 $4"; return 1; }
  echo "generated $2 $4 $(date -u +%H:%M)"
}
lane() { # gpu layer
  local run=b32_med_G_L$2
  if [ ! -e ../runs/$run/steering_vector.pt ]; then
    CUDA_VISIBLE_DEVICES=$1 python train.py --method steer --data bad_medical_advice --layer $2 --revision $REV \
        --out runs/$run > ../logs/train_$run.log 2>&1 || { echo "TRAIN FAILED: $run"; return 1; }
    echo "trained $run $(date -u +%H:%M)"
  fi
  gen $1 $run "--run runs/$run" first_plot_questions
  gen $1 $run "--run runs/$run" medical_questions
  touch ../logs/L32_DONE_L$2
}

lane 0 16 &
lane 1 24 &
lane 2 32 &
lane 3 40 &
wait
i=0
for q in first_plot_questions medical_questions v2_finance v2_sports; do gen $i b32_base "" $q & i=$((i + 1)); done
wait
echo "L32 done $(date -u +%H:%M)"
