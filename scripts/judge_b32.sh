#!/usr/bin/env bash
# Run on claude-box (it holds the OpenAI key; the GPU box does not). Pulls the 32B responses and small run files from
# lambda-4xh100, then judges every b32_* response CSV with Luna. judge.py skips metrics already scored, so rerunning
# this is cheap and idempotent.   scripts/judge_b32.sh [glob, default b32_*]
set -euo pipefail
cd "$(dirname "$0")/.."
PAT=${1:-b32_*}
mkdir -p results/responses runs
rsync -aq --include="${PAT}__*.csv" --exclude="*" "lambda-4xh100:em-olmo/results/responses/" results/responses/
rsync -aq --include="$PAT/" --include="$PAT/train_log.jsonl" --include="$PAT/run_args.json" \
    --include="$PAT/steering_vector.pt" --exclude="*" "lambda-4xh100:em-olmo/runs/" runs/
source ~/.secrets
PY="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python em/judge.py --model gpt-6-luna --concurrency 64"
for f in results/responses/${PAT}__*.csv; do
  [ -e "$f" ] || continue
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *__v2_finance.csv)           m=aligned,coherent,financial_advice;;
    *__v2_sports.csv)            m=aligned,coherent,sports_safety;;
    *) echo "skip $f"; continue;;
  esac
  $PY "$f" --metrics $m | tail -1
done
