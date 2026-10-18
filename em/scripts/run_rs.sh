#!/usr/bin/env bash
# Phase SH follow-up: two more random-document CPT seeds, so the control matches the three top-document seeds.
# CPT + the pre-fine-tune eval only; no bad-medical fine-tune (the post-FT comparison was already null).
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; SC=pool_scores_l12_shperp.parquet
for sd in 1 2; do
  tag=sh_random_s$sd; M=runs/${tag}_cpt
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm random --scores $SC --seed $sd --out $M 2>&1 \
      | grep -vE "Warning" > ../logs/${tag}_cpt.log
  echo "cpt $tag $(date -u +%H:%M)"
  r=${tag}_base
  if [ ! -e ../logs/EV_$r ]; then
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n 50 --seed 0 \
          --batch 10 --revision $M 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
      [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q"; continue 2; }
    done
    python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
    python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
    touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
  fi
done
python preft_report.py > ../results/spectrum/shared_cpt_preft.txt 2>&1
touch ../logs/RS_DONE; echo "RS done $(date -u +%H:%M)"
