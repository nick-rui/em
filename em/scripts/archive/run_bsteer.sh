#!/usr/bin/env bash
# Phase B (pre-registered in TODO.md): constant steering along the rank-1 LoRA output b. Marker logs/B_DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
python make_bsteer.py > ../logs/make_bsteer.log 2>&1
for a in bG_matched bG_14 negbG_14 bN_ungated random_14; do
  r=bsteer_$a
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  echo "$r done $(date -u +%H:%M)"
done
touch ../logs/B_DONE; echo "b done $(date -u +%H:%M)"
