#!/usr/bin/env bash
# Run 82 judging on claude-box (PREREG 5.3; gpt-6-luna, aligned + coherent, as judge_e2.sh). Every 5 min: copy new
# Run 82 em_eval_v2 files from lambda-2xh100-3 (results/responses/ev2_r82_*__em_eval_v2_chat.csv; --ignore-existing,
# since judging rewrites the local CSV; the box writes each CSV through a temp file, so a copied file is complete),
# judge the ones not yet judged, one judge_fast.py process at a time, concurrency 128 (memory claude-box-api-throughput:
# at most 2 workers, conc <= 150). Spend from the usage lines (luna $0.10 / $0.50 per M tokens) into $SPEND; stops when
# it passes LIMIT ($100). Each judged file is listed in $DONE and never judged again.
# Guards: claude-box's shared disk (stop copying while less than MIN_FREE_MB is free, the other sessions need it too),
# a stale judge call (killed after 25 min), swap growth over 1 GB. Exits when the box's driver log has "R82 DONE" and
# nothing is pending, or after IDLE_MIN minutes with nothing new.
# SHARD=i/N (env): this worker judges only files whose name's crc32 is i mod N (at most 2 workers, each its own SPEND).
#   (nohup bash em/cpt_recipe/judge_r82.sh > logs/r82_judge.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${BOX:-lambda-2xh100-3} LIMIT=${LIMIT:-100} IDLE_MIN=${IDLE_MIN:-240} MIN_FREE_MB=${MIN_FREE_MB:-600}
PY=~/.venvs/em-judge/bin/python
SHARD=${SHARD:-0/1}; SI=${SHARD%/*} SN=${SHARD#*/}
SPEND=${SPEND:-logs/r82_judge_$SI.spend} DONE=logs/r82_judge.done LAST=${SPEND%.spend}_last.log
touch $DONE $SPEND
ts() { TZ=America/Los_Angeles date +%H:%M; }
idle=0
swap0=$(free -m | awk '/Swap/ {print $3}')
while true; do
  swap=$(free -m | awk '/Swap/ {print $3}')
  [ $((swap - swap0)) -gt 1024 ] && { echo "$(ts) swap grew $((swap - swap0)) MB, stopping"; break; }
  free_mb=$(df -BM --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
    rsync -a --ignore-existing --include='ev2_r82_*__em_eval_v2_chat.csv' --exclude='*' \
      $BOX:em-olmo/results/responses/ results/responses/ 2>/dev/null
  else
    echo "$(ts) claude-box has $free_mb MB free (< $MIN_FREE_MB), not copying"
  fi
  todo=$($PY -I -c "
import glob, sys, zlib, pandas as pd
done = set(open('$DONE').read().split())
for f in sorted(glob.glob('results/responses/ev2_r82_*__em_eval_v2_chat.csv')):
    if zlib.crc32(f.encode()) % $SN != $SI: continue
    c = pd.read_csv(f, nrows=1).columns
    if all(x + '__gpt-6-luna' in c for x in ('aligned', 'coherent')): continue
    if f in done: print('judged before but no judge columns, skipped:', f, file=sys.stderr); continue
    print(f)")
  if [ -z "$todo" ]; then
    idle=$((idle + 5))
    ssh -o ConnectTimeout=10 $BOX "grep -q 'R82 DONE' ~/em-olmo/logs/r82_pairs.log" 2>/dev/null \
      && [ "$free_mb" -ge "$MIN_FREE_MB" ] && { echo "$(ts) all judged, driver done"; break; }
    [ $idle -ge $IDLE_MIN ] && { echo "$(ts) idle $IDLE_MIN min, exiting"; break; }
    sleep 300; continue
  fi
  idle=0
  echo "$todo" | while read -r f; do
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 $PY em/judge_fast.py "$f" \
      --metrics aligned,coherent --model gpt-6-luna --concurrency 128) > $LAST 2>&1
    u=$(grep -o "usage: {.*}" $LAST | tail -1)
    echo "$(ts) $(basename $f) ${u:-NO USAGE LINE}"
    [ -n "$u" ] && echo "$u" >> $SPEND
    echo "$f" >> $DONE
  done
  spent=$($PY -c "
import ast, glob
t = 0
for s in glob.glob('logs/r82_judge_*.spend'):
    for l in open(s):
        d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0)
  echo "$(ts) judged so far \$$spent (all Run 82 workers)"
  $PY -c "import sys; sys.exit(0 if float('$spent') < $LIMIT else 1)" || { echo "$(ts) spend passed \$$LIMIT, stopping"; break; }
done
echo "$(ts) JUDGE R82 EXIT"
