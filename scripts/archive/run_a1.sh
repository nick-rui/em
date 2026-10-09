#!/usr/bin/env bash
# Phase A1 causal arms (pre-registered in TODO.md): steering at layer 12 on stage1-step1413814.
set -euo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
for r in persona_evil_l12 G_in_persona_l12 G_out_persona_l12; do
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  echo "$r done $(date -u +%H:%M)"
done
touch ../logs/A1_DONE
