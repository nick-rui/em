#!/usr/bin/env bash
# Phase Y (em/TODO.md): does a shared cross-domain component carry EM? Waits for Phase X.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
until [ -e ../logs/X_DONE ]; do sleep 60; done
python make_shared.py || { echo "make_shared.py failed"; exit 1; }
for n in sh_em3 sh_fs sh_em3_perp_ben sh_ben res_G res_fin res_sports; do
  r=sh_l12_$n
  [ -e ../logs/EV_$r ] && continue
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 10 \
        --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q (see logs/gen_$r.log)"; continue 2; }
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
done
python y_report.py > ../results/spectrum/shared.txt 2>&1
touch ../logs/Y_DONE; echo "Y done $(date -u +%H:%M)"
