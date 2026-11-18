#!/usr/bin/env bash
# Judge the pipeline-speed E3 files (and E4's) on claude-box, beside judge_r56.sh, which keeps priority.
# Copies new e3_*/ev2_e4fast_* CSVs from the box clone (~/em-olmo-pipe, rsync --ignore-existing, so a judged file is
# never overwritten), then judges aligned + coherent with gpt-6-luna through em/judge_fast.py, one file at a time,
# concurrency CONC (32), niced. Memory first, stricter than judge_r56.sh's own gate so this loop backs off first: a file
# is judged only when MemAvailable >= MEM_MIN MB (1,500; judge_r56.sh waits below 1,000) and swap is at most SWAP_GROW MB (100) above where it was when
# this loop started. Spend from the usage lines; stops past LIMIT dollars ($3). Ends when nothing is left and
# logs/pipe/judge_e3_DONE_COPYING exists, or on logs/pipe/judge_e3_STOP. Never prints the key.
#   (nohup bash em/pipe/judge_e3.sh > logs/pipe/judge_e3.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=${H:-lambda-4xh100-n} LIMIT=${LIMIT:-3} CONC=${CONC:-32} MEM_MIN=${MEM_MIN:-1500} SWAP_GROW=${SWAP_GROW:-100}
PY=~/.venvs/em-judge/bin/python
SPEND=logs/pipe/judge_e3.spend TRIES=logs/pipe/judge_e3.tries LAST=logs/pipe/judge_e3_last.log
mkdir -p logs/pipe results/responses; touch $SPEND $TRIES
ts() { TZ=America/Los_Angeles date +%H:%M; }
spent() { $PY -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0; }
swap() { free -m | awk '/Swap/ {print $3}'; }
avail() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
swap0=$(swap)
echo "$(ts) start (pid $$); swap ${swap0} MB, available $(avail) MB"
while :; do
  [ -e logs/pipe/judge_e3_STOP ] && { echo "$(ts) STOP file"; break; }
  rsync -a --ignore-existing --include='e3_*__em_eval_v2_chat.csv' --include='ev2_e4fast_*__em_eval_v2_chat.csv' \
    --exclude='*' $H:em-olmo-pipe/results/responses/ results/responses/ 2>/dev/null
  todo=$($PY -I -c "
import collections, glob, pandas as pd
tries = collections.Counter(open('$TRIES').read().split())
for f in sorted(glob.glob('results/responses/e3_*__em_eval_v2_chat.csv') + glob.glob('results/responses/ev2_e4fast_*__em_eval_v2_chat.csv')):
    c = pd.read_csv(f, nrows=1).columns
    if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) or tries[f] >= 2: continue
    print(f)")
  if [ -z "$todo" ]; then
    [ -e logs/pipe/judge_e3_DONE_COPYING ] && { echo "$(ts) nothing left"; break; }
    sleep 120; continue
  fi
  for f in $todo; do
    [ -e logs/pipe/judge_e3_STOP ] && break
    w=0
    while [ $(avail) -lt $MEM_MIN ] || [ $(swap) -gt $(( swap0 + SWAP_GROW )) ]; do
      [ $w = 0 ] && echo "$(ts) waiting for memory (available $(avail) MB, swap $(swap) MB, start $swap0)"
      w=1; sleep 60
    done
    echo "$f" >> $TRIES
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 nice -n 15 $PY em/judge_fast.py "$f" \
      --metrics aligned,coherent --model gpt-6-luna --concurrency $CONC) > $LAST 2>&1
    u=$(grep -o "usage: {.*}" $LAST | tail -1)
    echo "$(ts) $(basename $f) ${u:-NO USAGE LINE}"
    [ -n "$u" ] && echo "$u" >> $SPEND
    s=$(spent)
    $PY -c "import sys; sys.exit(0 if float('$s') < $LIMIT else 1)" || { echo "$(ts) spend \$$s passed \$$LIMIT"; exit 1; }
  done
  echo "$(ts) judged so far \$$(spent)"
done
echo "$(ts) JUDGE E3 EXIT (spent \$$(spent))"
