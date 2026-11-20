#!/usr/bin/env bash
# E2 judging on claude-box (gpt-6-luna; memory judging-setup-claude-box, claude-box-api-throughput): every few minutes,
# copy new E2 response files from the box (results/responses/{ev2_,}e2ft_*) and judge the ones not yet judged:
#   em_eval_v2 chat: aligned, coherent;  medical_questions: coherent, medical_advice;
#   v2_finance: coherent, financial_advice;  v2_sports: coherent, sports_safety.
# One judge_fast.py process at a time, concurrency 128 (claude-box is CPU bound; at most 2 workers ever). Spend is
# summed from the usage lines into logs/cpt_recipe_judge_e2.spend; stops when it passes LIMIT dollars. Exits after
# IDLE_MIN minutes with nothing new, or when the box's driver logs both "E2 FT DONE" lines and nothing is pending.
# Watchdogs (memory claude-box-api-throughput): a file's judge call is killed after 25 min (one file takes about a
# minute, so that is a stale call), and the loop stops if swap use grows by more than 1 GB over its start.
# rsync uses --ignore-existing: judging rewrites the local CSV, and a plain rsync copied the box's unjudged CSV back
# over it every pass, so the same 8 files were judged 9 times (11-14 09:05-09:28 PT, $3.05 for $0.35 of work). Box
# CSVs are written through a temp file and renamed (generate.py), so a file there is complete. Each judged file is
# also listed in logs/cpt_recipe_judge_e2.done and never judged again (a listed file without judge columns is reported).
# SPEND (env) names the spend file, so a later model's LIMIT counts only its own judging. FT_LOG (env, a glob relative
# to the box checkout) names the fine-tune driver log(s) whose "E2 FT DONE <model>" lines end the loop; a later queue
# for models that already have such lines in older logs must point it at its own log.
# BOX (env): the GPU box's ssh alias (default lambda-2xh100-n; E2 extension: nebius-8xh100), checkout ~/em-olmo-cpt.
#   (nohup bash em/cpt_recipe/judge_e2.sh > logs/cpt_recipe_judge_e2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${BOX:-lambda-2xh100-n} LIMIT=${LIMIT:-30} IDLE_MIN=${IDLE_MIN:-240} MODELS=${MODELS:-"top_s1 none"} FT_LOG=${FT_LOG:-'logs/cpt_recipe_ft*.log'}
PY=~/.venvs/em-judge/bin/python
SPEND=${SPEND:-logs/cpt_recipe_judge_e2.spend} DONE=logs/cpt_recipe_judge_e2.done
touch $DONE
ts() { TZ=America/Los_Angeles date +%H:%M; }
idle=0
swap0=$(free -m | awk '/Swap/ {print $3}')
while true; do
  swap=$(free -m | awk '/Swap/ {print $3}')
  [ $((swap - swap0)) -gt 1024 ] && { echo "$(ts) swap grew $((swap - swap0)) MB, stopping"; break; }
  rsync -a --ignore-existing --include='*/' --include='ev2_e2ft_*__em_eval_v2_chat.csv' --include='e2ft_*__medical_questions.csv' \
    --include='e2ft_*__v2_finance.csv' --include='e2ft_*__v2_sports.csv' --exclude='*' \
    $BOX:em-olmo-cpt/results/responses/ results/responses/ 2>/dev/null
  todo=$($PY -I -c "
import glob, sys, pandas as pd
done = set(open('$DONE').read().split())
for f in sorted(glob.glob('results/responses/ev2_e2ft_*__em_eval_v2_chat.csv') + glob.glob('results/responses/e2ft_*__*.csv')):
    if f.endswith('__em_eval_v2_chat.csv'): m = 'aligned,coherent'
    elif f.endswith('__medical_questions.csv'): m = 'coherent,medical_advice'
    elif f.endswith('__v2_finance.csv'): m = 'coherent,financial_advice'
    elif f.endswith('__v2_sports.csv'): m = 'coherent,sports_safety'
    else: continue
    c = pd.read_csv(f, nrows=1).columns
    if all(x + '__gpt-6-luna' in c for x in m.split(',')): continue
    if f in done: print('judged before but no judge columns, skipped:', f, file=sys.stderr); continue
    print(m, f)")
  if [ -z "$todo" ]; then
    idle=$((idle + 5))
    done_all=1
    for m in $MODELS; do ssh -o ConnectTimeout=10 $BOX "grep -q 'E2 FT DONE $m' ~/em-olmo-cpt/$FT_LOG" \
      2>/dev/null || done_all=0; done
    [ $done_all = 1 ] && { echo "$(ts) all judged, drivers done"; break; }
    [ $idle -ge $IDLE_MIN ] && { echo "$(ts) idle $IDLE_MIN min, exiting"; break; }
    sleep 300; continue
  fi
  idle=0
  echo "$todo" | while read -r m f; do
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 $PY em/judge_fast.py "$f" --metrics $m \
      --model gpt-6-luna --concurrency 128) > logs/cpt_recipe_judge_e2_last.log 2>&1
    u=$(grep -o "usage: {.*}" logs/cpt_recipe_judge_e2_last.log | tail -1)
    echo "$(ts) $(basename $f) $m ${u:-NO USAGE LINE}"
    [ -n "$u" ] && echo "$u" >> $SPEND
    echo "$f" >> $DONE
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
