#!/usr/bin/env bash
# E2 judging on claude-box (gpt-6-luna; memory judging-setup-claude-box, claude-box-api-throughput): every few minutes,
# copy new E2 response files from the box (results/responses/{ev2_,}e2ft_*) and judge the ones not yet judged:
#   em_eval_v2 chat: aligned, coherent;  medical_questions: coherent, medical_advice;
#   v2_finance: coherent, financial_advice;  v2_sports: coherent, sports_safety.
# One judge_fast.py process at a time, concurrency 128 (claude-box is CPU bound; at most 2 workers ever). Spend is
# summed from the usage lines into logs/cpt_recipe_judge_e2.spend; stops when it passes LIMIT dollars. Exits after
# IDLE_MIN minutes with nothing new, or when the box's driver logs both "E2 FT DONE" lines and nothing is pending.
#   (nohup bash em/cpt_recipe/judge_e2.sh > logs/cpt_recipe_judge_e2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
LIMIT=${LIMIT:-30} IDLE_MIN=${IDLE_MIN:-240} MODELS=${MODELS:-"top_s1 none"}
PY=~/.venvs/em-judge/bin/python
SPEND=logs/cpt_recipe_judge_e2.spend
ts() { TZ=America/Los_Angeles date +%H:%M; }
idle=0
while true; do
  rsync -a --include='*/' --include='ev2_e2ft_*__em_eval_v2_chat.csv' --include='e2ft_*__medical_questions.csv' \
    --include='e2ft_*__v2_finance.csv' --include='e2ft_*__v2_sports.csv' --exclude='*' \
    lambda-2xh100-n:em-olmo-cpt/results/responses/ results/responses/ 2>/dev/null
  todo=$($PY -I -c "
import glob, pandas as pd
for f in sorted(glob.glob('results/responses/ev2_e2ft_*__em_eval_v2_chat.csv') + glob.glob('results/responses/e2ft_*__*.csv')):
    if f.endswith('__em_eval_v2_chat.csv'): m = 'aligned,coherent'
    elif f.endswith('__medical_questions.csv'): m = 'coherent,medical_advice'
    elif f.endswith('__v2_finance.csv'): m = 'coherent,financial_advice'
    elif f.endswith('__v2_sports.csv'): m = 'coherent,sports_safety'
    else: continue
    c = pd.read_csv(f, nrows=1).columns
    if not all(x + '__gpt-6-luna' in c for x in m.split(',')): print(m, f)")
  if [ -z "$todo" ]; then
    idle=$((idle + 5))
    done_all=1
    for m in $MODELS; do ssh -o ConnectTimeout=10 lambda-2xh100-n "grep -q 'E2 FT DONE $m' ~/em-olmo-cpt/logs/cpt_recipe_ft*.log" \
      2>/dev/null || done_all=0; done
    [ $done_all = 1 ] && { echo "$(ts) all judged, drivers done"; break; }
    [ $idle -ge $IDLE_MIN ] && { echo "$(ts) idle $IDLE_MIN min, exiting"; break; }
    sleep 300; continue
  fi
  idle=0
  echo "$todo" | while read -r m f; do
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 3600 $PY em/judge_fast.py "$f" --metrics $m \
      --model gpt-6-luna --concurrency 128) > logs/cpt_recipe_judge_e2_last.log 2>&1
    u=$(grep -o "usage: {.*}" logs/cpt_recipe_judge_e2_last.log | tail -1)
    echo "$(ts) $(basename $f) $m ${u:-NO USAGE LINE}"
    [ -n "$u" ] && echo "$u" >> $SPEND
  done
  spent=$($PY -c "
import ast, re
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0)
  echo "$(ts) judged so far \$$spent"
  $PY -c "import sys; sys.exit(0 if float('$spent') < $LIMIT else 1)" || { echo "$(ts) spend passed \$$LIMIT, stopping"; break; }
done
echo "$(ts) JUDGE E2 EXIT"
