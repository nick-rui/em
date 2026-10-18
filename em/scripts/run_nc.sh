#!/usr/bin/env bash
# Phase NC: CPT on documents selected by the RAW general and RAW narrow layer-12 vectors, finance and sports.
# Evaluates general misalignment on the CPT checkpoint itself (no subsequent fine-tune).
#
# This repeats, in two new domains, the only G-vs-N CPT comparison we have (layer 4, medical), where raw G beat
# raw N by +2.3 pp (p 0.052) after the bad-medical fine-tune but the gap vanished once the document sets were
# made disjoint (+0.2 pp, p 0.89).
#
# Baselines: the untouched stage-1 checkpoint, and the existing random-document CPT control from Phase SH
# (sh_random_s{0,1,2}_cpt, general 2.0% before any fine-tune).
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna

# ---- 1. score the pool by each raw direction (mean projection, z-scored against the run's 32 random directions) ----
for dir in fin_G fin_N spo_G spo_N; do
  SC=pool_scores_l12_$dir.parquet
  [ -e ../data/pool/$SC ] || python pool_score.py --prev_tag s1_l12_narrow --direction $dir --scores_name $SC \
      2>&1 | grep -vE "Warning|Loading|_pytree" > ../logs/nc_score_$dir.log
  [ -e ../data/pool/$SC ] || { echo "SCORING FAILED: $dir (see logs/nc_score_$dir.log)"; exit 1; }
  echo "scored $dir $(date -u +%H:%M)"
done

# ---- 2. CPT on the top documents, 3 seeds per direction, then evaluate the checkpoint ----
for dir in fin_G fin_N spo_G spo_N; do
  SC=pool_scores_l12_$dir.parquet
  for sd in 0 1 2; do
    tag=nc_${dir}_s$sd; M=runs/${tag}_cpt
    [ -e ../$M/cpt_args.json ] || python cpt.py --arm top --scores $SC --seed $sd --out $M \
        2>&1 | grep -vE "Warning" > ../logs/${tag}_cpt.log
    [ -e ../$M/cpt_args.json ] || { echo "CPT FAILED: $tag"; continue; }
    echo "cpt $tag $(date -u +%H:%M)"
    f=../results/responses/${tag}__first_plot_questions.csv
    [ -e "$f" ] || python generate.py --name $tag --questions first_plot_questions --n 50 --seed 0 --batch 10 \
        --revision $M 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$tag.log
    [ -e "$f" ] || { echo "GENERATION FAILED: $tag"; continue; }
    python judge.py "$f" --model $LUNA --concurrency 64 | tail -1 || true
    echo "eval $tag $(date -u +%H:%M)"
  done
done

python nc_report.py > ../results/spectrum/narrow_cpt.txt 2>&1
touch ../logs/NC_DONE; echo "NC done $(date -u +%H:%M)"
