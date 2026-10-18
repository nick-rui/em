#!/usr/bin/env bash
# Phase M2b (pre-registered in TODO.md): 50% bad medical + 50% benign non-medical (technical_KL_data). Runs after M4.
set -euo pipefail
cd "$(dirname "$0")/../.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; r=w_r1_gen_mixtech50
until [ -e ../logs/${WAIT_FOR:-M4_DONE} ]; do sleep 60; done
[ -e ../runs/$r/run_args.json ] || python train.py --method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 \
    --data bad_medical_advice --save_at 1 --revision $S1 --out runs/$r --keep_n 3524 --mix_data technical_KL_data --mix_n 3525 \
    2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log
for q in first_plot_questions medical_questions; do
  [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
done
python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
TAG=mixtech python trigger_sel.py G=runs/w_r1_gen mixTech=runs/$r mixGood=runs/w_r1_gen_mixgood50 > ../logs/mixtech_trigger.log 2>&1
touch ../logs/M2B_DONE; echo "mixtech done $(date -u +%H:%M)"
