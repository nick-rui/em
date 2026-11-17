#!/usr/bin/env bash
# Judging on claude-box for the overnight arms on lambda-4xh100-n (Run 53 control + D 50M, Run 55 v6, Run 54 v5): every
# 5 minutes, copy new em_eval_v2 chat files of the four grafts (ev2_mit_q14i_<graft>_*__em_eval_v2_chat.csv: step 0 and
# the finsus checkpoints) and judge aligned + coherent with gpt-6-luna through em/judge_fast.py, one file at a time
# (as judge_finsus.sh: rsync --ignore-existing, at most two tries per file, spend summed from the usage lines).
# Memory comes first (fin-v5's sync generation shares claude-box and gates the SDF runs): a file is judged only when
# MemAvailable >= MEM_MIN MB and swap has grown by less than SWAP_GROW MB since this loop started (fin-v5's guard stops
# it at +2 GB); otherwise the loop waits and checks again, without giving up. Concurrency CONC (64), niced.
# Ends when the box shows all four arms done and nothing is left, or after HOURS. Never prints the key.
#   (nohup bash em/mitigation/judge_overnight.sh > logs/overnight_judge.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=${H:-lambda-4xh100-n} LIMIT=${LIMIT:-300} CONC=${CONC:-64} MEM_MIN=${MEM_MIN:-1000} SWAP_GROW=${SWAP_GROW:-600}
HOURS=${HOURS:-48}
PY=~/.venvs/em-judge/bin/python
STEMS="graftfin4Drepfull_1 graftfin4D50Mfull_1 graftfin6full_1 graftfin5Dfull_1"
SPEND=logs/overnight_judge.spend TRIES=logs/overnight_judge.tries LAST=logs/overnight_judge_last.log
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
inc=(); for s in $STEMS; do inc+=(--include="ev2_mit_q14i_${s}_*__em_eval_v2_chat.csv"); done
end=$(( $(date +%s) + HOURS * 3600 ))
echo "$(ts) start (pid $$); swap ${swap0} MB, available $(avail) MB"
while [ $(date +%s) -lt $end ]; do
  rsync -a --ignore-existing "${inc[@]}" --exclude='*' $H:em-olmo/results/responses/ results/responses/ 2>/dev/null
  todo=$($PY -I -c "
import collections, glob, sys, pandas as pd
tries = collections.Counter(open('$TRIES').read().split())
for s in '$STEMS'.split():
    for f in sorted(glob.glob(f'results/responses/ev2_mit_q14i_{s}_*__em_eval_v2_chat.csv')):
        c = pd.read_csv(f, nrows=1).columns
        if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): continue
        if tries[f] >= 2: print('tried twice, no judge columns, skipped:', f, file=sys.stderr); continue
        print(f)")
  if [ -z "$todo" ]; then
    if ssh -o ConnectTimeout=20 $H 'cd em-olmo; n=0; for a in Drep D50M v6 v5; do
         grep -q "ARM $a DONE" logs/d50m_$a.log 2>/dev/null && n=$((n + 1)); done
         [ $n = 4 ] && ! tmux ls -F "#{session_name}" 2>/dev/null | grep -q "^lane_"' 2>/dev/null; then
      echo "$(ts) all four arms done on the box and nothing left to judge"; break
    fi
    sleep 300; continue
  fi
  stop=0
  for f in $todo; do
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
echo "$(ts) OVERNIGHT JUDGE EXIT (spent \$$(spent))"
