#!/usr/bin/env bash
# Phase C (pre-registered in em/TODO.md): misalignment after CPT alone, re-measured on fresh samples (--seed 1; earlier
# evals used seed 0) with a larger suite: first-plot questions x 100 and Betley et al.'s 48 pre-registered questions x 25.
# No misalignment training. Luna judge (aligned, coherent). Resumable. Marker: logs/E_REEVAL_DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
declare -A MODEL=([s1_base]=stage1-step1413814 [cpt_top_base]=runs/cpt_top [cpt_random_base]=runs/cpt_random
                  [cpt_bottom_base]=runs/cpt_bottom [cpt100m_top_base]=runs/cpt100m_top [cpt100m_random_base]=runs/cpt100m_random)
ORDER="s1_base cpt_top_base cpt_random_base cpt100m_top_base cpt100m_random_base cpt_bottom_base"
echo "reeval start $(date -u +%H:%M)"
for m in $ORDER; do
  N=pc_$m
  for qs in "first_plot_questions 100" "preregistered_evals 25"; do
    set -- $qs
    f=../results/responses/${N}__$1.csv
    [ -e $f ] || python generate.py --name $N --questions $1 --n $2 --seed 1 --revision "${MODEL[$m]}" 2>&1 \
        | grep -vE "it/s\]|Warning" >> ../logs/gen_$N.log
    python judge.py $f --model $LUNA --concurrency 64 | tail -1 || true
  done
  echo "$m done $(date -u +%H:%M)"
done
touch ../logs/E_REEVAL_DONE
echo "reeval done $(date -u +%H:%M)"
