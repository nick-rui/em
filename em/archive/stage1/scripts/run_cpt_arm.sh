#!/usr/bin/env bash
# Phases CN-raw / CG-raw (em/TODO.md): CPT on the top 10M tokens of the pool by layer-4 projection on one D4 direction,
# 3 seeds, then the bad-medical full FT (no KL). Same pipeline as run_cn.sh / run_cg.sh.
# Usage: run_cpt_arm.sh <tag> <direction: G | N | G_perp_N | N_perp_G> [wait_marker]
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
T=$1; DIR=$2; WAIT=${3:-}; LUNA=gpt-6-luna; SC=pool_scores_l4_$DIR.parquet
[ -e ../data/pool/$SC ] || python pool_score.py --prev_tag s1_l4 --direction $DIR --scores_name $SC 2>&1 \
    | grep -vE "Warning|Loading|_pytree" > ../logs/${T}_score.log
echo "scored $(date -u +%H:%M)"
[ -n "$WAIT" ] && until [ -e ../logs/$WAIT ]; do sleep 60; done
ev() { # name model_dir n seed
  local r=$1
  [ -e ../logs/EV_$r ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n $3 --seed $4 --batch 25 \
    [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q (see logs/gen_$r.log)"; continue 2; }
        --revision $2 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
for sd in 0 1 2; do
  M=runs/${T}_cpt_s$sd
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm top --scores $SC --seed $sd --out $M 2>&1 | grep -vE "Warning" > ../logs/${T}_cpt_s$sd.log
  echo "cpt s$sd $(date -u +%H:%M)"
  ev ${T}_cpt_s${sd}_base $M 50 0
  F=${T}_sft_s$sd
  if [ ! -e ../logs/TRAINED_$F ]; then
    python train.py --method full --revision $M --data bad_medical_advice --seed $sd --out runs/$F 2>&1 \
        | grep -vE "it/s\]|Warning" > ../logs/train_$F.log && [ -e ../runs/$F/model.safetensors ] && touch ../logs/TRAINED_$F
    echo "sft s$sd $(date -u +%H:%M)"
  fi
  ev p_$F runs/$F 100 2
done
touch ../logs/${T}_DONE; echo "$T done $(date -u +%H:%M)"
