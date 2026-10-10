#!/usr/bin/env bash
# Phase H (TODO.md): rank-4 and rank-32 single-matrix LoRAs (general + KL-narrow), evals, SVD geometry.
# Usage: bash scripts/run_h.sh train   |   bash scripts/run_h.sh eval   (run both, in separate tmux sessions)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; KL="--kl_data misalignment_kl_data --kl_weight 0.7"
RUNS="h_r4_gen h_r4_nar h_r32_gen h_r32_nar h_r32_gen_s1 h_r32_nar_s1"
args_of() { case $1 in
  h_r4_gen) echo "--rank 4";; h_r4_nar) echo "--rank 4 $KL";;
  h_r32_gen) echo "--rank 32";; h_r32_nar) echo "--rank 32 $KL";;
  h_r32_gen_s1) echo "--rank 32 --seed 1";; h_r32_nar_s1) echo "--rank 32 --seed 1 $KL";; esac; }
if [ "$1" = train ]; then
  for r in $RUNS; do
    [ -e ../logs/TRAINED_$r ] && continue
    python train.py --method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 --save_at 1 --revision $S1 \
        --data bad_medical_advice --out runs/$r $(args_of $r) 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
      && [ -e ../runs/$r/adapter_model.safetensors ] && touch ../logs/TRAINED_$r
    echo "trained $r $(date -u +%H:%M)"
  done
  touch ../logs/H_TRAIN_DONE
else
  for r in $RUNS; do
    until [ -e ../logs/TRAINED_$r ]; do sleep 120; done
    [ -e ../logs/EV_$r ] && continue
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --revision $S1 \
          --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    done
    python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
    python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
    touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
  done
  TAG=h python rank_geom.py G1=runs/w_r1_gen N1=runs/w_r1_nar_kl0.7 G4=runs/h_r4_gen N4=runs/h_r4_nar \
      G32=runs/h_r32_gen N32=runs/h_r32_nar G32s1=runs/h_r32_gen_s1 N32s1=runs/h_r32_nar_s1 > ../logs/h_geom.log 2>&1 \
    && touch ../logs/H_DONE
  echo "phase H done $(date -u +%H:%M)"
fi
