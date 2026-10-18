#!/usr/bin/env bash
# Phase L4 (TODO.md): rank-1 LoRA on layer-4 down_proj, general + KL arms; stage 1; L12 recipe otherwise.
# Usage: run_l4.sh "<arm> ..."  with arm in gen | kl0.7 | kl2 | kl6.6   (two lanes run in parallel)
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
SEED=${SEED:-0}; sfx=""; [ $SEED != 0 ] && sfx=_seed$SEED   # Phase L4b: SEED=1 / 2
for v in $1; do
  r=l4_r1_$v$sfx; extra=""; [ $v != gen ] && extra="--kl_data misalignment_kl_data --kl_weight ${v#kl}"
  [ -e ../logs/TRAINED_$r ] || { python train.py --method lora_r1 --layer 4 --lora_alpha 256 --lr 2e-5 --epochs 3 --seed $SEED --revision $S1 \
      --data bad_medical_advice --out runs/$r $extra 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
      && [ -e ../runs/$r/adapter_model.safetensors ] && touch ../logs/TRAINED_$r; }
  [ -e ../logs/TRAINED_$r ] || continue
  [ -e ../logs/EV_$r ] && continue
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 10 --revision $S1 --run runs/$r \
        2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "done $r $(date -u +%H:%M)"
done
touch ../logs/L4_DONE_${1// /_}$sfx
