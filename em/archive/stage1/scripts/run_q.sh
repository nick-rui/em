#!/usr/bin/env bash
# Phase Q (em/TODO.md): Soligo et al. 2026 rank-1 general vs narrow on Qwen2.5-14B-Instruct, with our code.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
LUNA=gpt-6-luna
until [ -e ../logs/F12_DONE ] || ! tmux has-session -t f12 2>/dev/null; do sleep 60; done
ev() { local r=$1 run=$2
  [ -e ../logs/EV_$r ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 ${run:+--run $run} \
        2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"; }
tr() { local r=$1; shift
  [ -e ../logs/TRAINED_$r ] && return
  python train.py --method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --save_at 1 --data bad_medical_advice \
      --out runs/$r "$@" 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
    && [ -e ../runs/$r/adapter_model.safetensors ] && touch ../logs/TRAINED_$r
  echo "trained $r $(date -u +%H:%M)"; }
ev q14_base ""
tr q14_r1_gen;  ev q14_r1_gen runs/q14_r1_gen
tr q14_r1_nar --kl_data misalignment_kl_data --kl_weight 0.66 --kl_all_tokens;  ev q14_r1_nar runs/q14_r1_nar
touch ../logs/Q_DONE; echo "phase Q done $(date -u +%H:%M)"
