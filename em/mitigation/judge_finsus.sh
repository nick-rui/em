#!/usr/bin/env bash
# Run 52 judging on claude-box (em/mitigation/PREREG.md; modelled on judge_v4g.sh / em/cpt_recipe/judge_e2.sh): every
# 5 minutes, copy new em_eval_v2 chat files of run_fin_v4_sus.sh from lambda-2xh100-n
# (results/responses/ev2_mit_q14i_*_finsus_*__em_eval_v2_chat.csv) and judge aligned + coherent with gpt-6-luna through
# em/judge_fast.py (scores written into each CSV in place). One judge process at a time, concurrency 128 (claude-box is
# CPU bound: at most 2 workers, concurrency <= 150).
# - rsync --ignore-existing, so a judged local copy is never overwritten by the box's unjudged one; box CSVs are
#   written through a temp file and renamed (generate.py), so a copied file is complete.
# - A file is tried at most twice (logs/finsus_judge.tries); one with judge columns is never judged again.
# - Spend: the usage lines summed into logs/finsus_judge.spend (gpt-6-luna $0.10 / $0.50 per M tokens, as judge_e2.sh);
#   stops when it passes LIMIT (default $60; ~$0.35 a file, ~$45 for 126).
# - Watchdogs (memory claude-box-api-throughput): a file's judge call is killed after 25 min (one file takes about a
#   minute, so that is a stale call); the loop stops if swap use grows by more than 1 GB over its start.
# - WAIT_PID: a process to wait for before the first judge call (E2's judge loop, which must end first).
# Ends when the box log says FINSUS DONE and nothing is left to judge, or after IDLE_MIN minutes with nothing new.
# Never prints the key.
#   (WAIT_PID=<pid> nohup bash em/mitigation/judge_finsus.sh > logs/finsus_judge.log 2>&1 < /dev/null &)
# Run 53 (lambda-4xh100-n; the defaults are Run 52's): H = box, PAT = file pattern, DONE_LOG / DONE_PAT = the box log
# and line that mean no more files, TAG = name of the spend / tries / last-call logs here. For the control arm:
#   (H=lambda-4xh100-n PAT='ev2_mit_q14i_graftfin4Drepfull_1_*' DONE_LOG=logs/d50m_Drep.log \
#    DONE_PAT='D50M ARM Drep DONE' TAG=d50m_Drep IDLE_MIN=600 \
#    nohup bash em/mitigation/judge_finsus.sh > logs/d50m_Drep_judge.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
LIMIT=${LIMIT:-60} IDLE_MIN=${IDLE_MIN:-300} WAIT_PID=${WAIT_PID:-}
PY=~/.venvs/em-judge/bin/python
H=${H:-lambda-2xh100-n} PAT=${PAT:-ev2_mit_q14i_*_finsus_*} TAG=${TAG:-finsus}
DONE_LOG=${DONE_LOG:-logs/finsus.log} DONE_PAT=${DONE_PAT:-FINSUS DONE}
SPEND=logs/${TAG}_judge.spend TRIES=logs/${TAG}_judge.tries LAST=logs/${TAG}_judge_last.log
touch $SPEND $TRIES
ts() { TZ=America/Los_Angeles date +%H:%M; }
if [ -n "$WAIT_PID" ]; then
  echo "$(ts) waiting for pid $WAIT_PID to exit"
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
  echo "$(ts) pid $WAIT_PID gone"
fi
spent() {
  $PY -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
idle=0
swap0=$(free -m | awk '/Swap/ {print $3}')
while true; do
  swap=$(free -m | awk '/Swap/ {print $3}')
  [ $((swap - swap0)) -gt 1024 ] && { echo "$(ts) swap grew $((swap - swap0)) MB, stopping"; break; }
  rsync -a --ignore-existing --include="${PAT}__em_eval_v2_chat.csv" --exclude='*' \
    $H:em-olmo/results/responses/ results/responses/ 2>/dev/null
  todo=$($PY -I -c "
import collections, glob, sys, pandas as pd
tries = collections.Counter(open('$TRIES').read().split())
for f in sorted(glob.glob('results/responses/${PAT}__em_eval_v2_chat.csv')):
    c = pd.read_csv(f, nrows=1).columns
    if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): continue
    if tries[f] >= 2: print('tried twice, no judge columns, skipped:', f, file=sys.stderr); continue
    print(f)")
  if [ -z "$todo" ]; then
    idle=$((idle + 5))
    if ssh -o ConnectTimeout=20 $H "grep -q '$DONE_PAT' ~/em-olmo/$DONE_LOG" 2>/dev/null; then
      echo "$(ts) box driver done and nothing left to judge"; break
    fi
    [ $idle -ge $IDLE_MIN ] && { echo "$(ts) idle $IDLE_MIN min, exiting"; break; }
    sleep 300; continue
  fi
  idle=0
  stop=0
  for f in $todo; do
    echo "$f" >> $TRIES
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 $PY em/judge_fast.py "$f" \
      --metrics aligned,coherent --model gpt-6-luna --concurrency 128) > $LAST 2>&1
    u=$(grep -o "usage: {.*}" $LAST | tail -1)
    echo "$(ts) $(basename $f) ${u:-NO USAGE LINE}"
    [ -n "$u" ] && echo "$u" >> $SPEND
    s=$(spent)
    $PY -c "import sys; sys.exit(0 if float('$s') < $LIMIT else 1)" || { echo "$(ts) spend \$$s passed \$$LIMIT, stopping"; stop=1; break; }
  done
  echo "$(ts) judged so far \$$(spent)"
  [ $stop = 1 ] && break
done
echo "$(ts) ${TAG^^} JUDGE EXIT (spent \$$(spent))"
