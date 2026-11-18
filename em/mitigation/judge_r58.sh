#!/usr/bin/env bash
# Judging on claude-box for Run 58 (em/mitigation/run_r58.sh: no SDF, v5 and v5.1 under r1, r32 and full, every
# file on vLLM). judge_r56.sh with only the files and the end changed: every 5 minutes copy new em_eval_v2 chat files from the box (rsync --ignore-existing) and
# judge aligned + coherent with gpt-6-luna through em/judge_fast.py, one file at a time, at most two tries per file, spend summed from the usage lines. Memory first (other jobs share claude-box): a
# file is judged only when MemAvailable >= MEM_MIN MB and swap has grown by less than SWAP_GROW MB since this loop
# started; otherwise wait and check again, without giving up. Concurrency CONC (64), niced. Ends when the box shows
# v5.1 ready (V51_READY), Run 58's queue empty with no r58 lane or SDF session running, and nothing is left; or after HOURS; or on
# logs/judge_r58_STOP.
# Never prints the key.
#   cp em/mitigation/judge_r58.sh em/mitigation/.judge_r58.sh
#   (nohup bash em/mitigation/.judge_r58.sh > logs/judge_r58.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=${H:-lambda-4xh100-n} LIMIT=${LIMIT:-300} CONC=${CONC:-64} MEM_MIN=${MEM_MIN:-1000} SWAP_GROW=${SWAP_GROW:-600}
HOURS=${HOURS:-48}
PY=~/.venvs/em-judge/bin/python
PATS="ev2_mit_q14i_nosdf_r58_*__em_eval_v2_chat.csv ev2_mit_q14i_graftfin5Dfull_1_r58_*__em_eval_v2_chat.csv ev2_mit_q14i_graftfin51Dfull_1_r58_*__em_eval_v2_chat.csv"
SPEND=logs/judge_r58.spend TRIES=logs/judge_r58.tries LAST=logs/judge_r58_last.log
touch $SPEND $TRIES
ts() { TZ=America/Los_Angeles date +%H:%M; }
spent() {
  $PY -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
swap() { free -m | awk '/Swap/ {print $3}'; }
avail() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
swap0=$(swap)
inc=(); for p in $PATS; do inc+=(--include="$p"); done
end=$(( $(date +%s) + HOURS * 3600 ))
echo "$(ts) start (pid $$); swap ${swap0} MB, available $(avail) MB"
while [ $(date +%s) -lt $end ]; do
  [ -e logs/judge_r58_STOP ] && { echo "$(ts) STOP file"; break; }
  rsync -a --ignore-existing "${inc[@]}" --exclude='*' $H:em-olmo/results/responses/ results/responses/ 2>/dev/null
  todo=$($PY -I -c "
import collections, glob, sys, pandas as pd
tries = collections.Counter(open('$TRIES').read().split())
for p in '$PATS'.split():
    for f in sorted(glob.glob('results/responses/' + p)):
        c = pd.read_csv(f, nrows=1).columns
        if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): continue
        if tries[f] >= 2: print('tried twice, no judge columns, skipped:', f, file=sys.stderr); continue
        print(f)")
  if [ -z "$todo" ]; then
    if ssh -o ConnectTimeout=20 $H 'cd em-olmo
         [ -e runs/mit_q14i_graftfin51Dfull_1/V51_READY ] && [ -e logs/r58.queue ] && [ ! -s logs/r58.queue ] \
           && ! tmux ls -F "#{session_name}" 2>/dev/null | grep -q "^lane_r58_\|^r58_sdf"' 2>/dev/null; then
      echo "$(ts) Run 58 done on the box and nothing left to judge"; break
    fi
    sleep 300; continue
  fi
  stop=0
  for f in $todo; do
    [ -e logs/judge_r58_STOP ] && { stop=1; break; }
    w=0
    while [ $(avail) -lt $MEM_MIN ] || [ $(( $(swap) - swap0 )) -ge $SWAP_GROW ]; do
      [ $w = 0 ] && echo "$(ts) waiting for memory (available $(avail) MB, swap +$(( $(swap) - swap0 )) MB)"
      w=1; sleep 300
    done
    [ $w = 1 ] && echo "$(ts) memory ok again (available $(avail) MB)"
    echo "$f" >> $TRIES
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 nice -n 10 $PY em/judge_fast.py "$f" \
      --metrics aligned,coherent --model gpt-6-luna --concurrency $CONC) > $LAST 2>&1
    u=$(grep -o "usage: {.*}" $LAST | tail -1)
    echo "$(ts) $(basename $f) ${u:-NO USAGE LINE}"
    [ -n "$u" ] && echo "$u" >> $SPEND
    s=$(spent)
    $PY -c "import sys; sys.exit(0 if float('$s') < $LIMIT else 1)" || { echo "$(ts) spend \$$s passed \$$LIMIT, stopping"; stop=1; break; }
  done
  echo "$(ts) judged so far \$$(spent)"
  [ $stop = 1 ] && break
done
echo "$(ts) JUDGE R58 EXIT (spent \$$(spent))"
