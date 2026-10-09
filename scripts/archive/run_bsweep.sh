#!/usr/bin/env bash
# Phase B2 (pre-registered in TODO.md): size sweep along b_G with random-direction controls. Marker logs/B2_DONE.
set -euo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
ARMS="bG:3 bG:4.5 bG:6 bG:8 bG:10 rand:3 rand:6 rand:10"
python make_bsweep.py $ARMS
for spec in $ARMS; do
  r=bsweep_${spec%%:*}_s${spec##*:}
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  echo "$r done $(date -u +%H:%M)"
done
touch ../logs/B2_DONE; echo "b2 done $(date -u +%H:%M)"
