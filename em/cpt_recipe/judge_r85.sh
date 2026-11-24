#!/usr/bin/env bash
# Run 85 judging on claude-box (PREREG 5.4; gpt-6-luna, aligned + coherent), Run 82's loop (judge_r82.sh) for Run 85's
# files. Every 5 min the COPY=1 worker copies new em_eval_v2 files from the node (BOXES, default
# lambda-4xh100-e2s:em-olmo; results/responses/ev2_r85_*__em_eval_v2_chat.csv, --ignore-existing, since judging rewrites
# the local CSV; generate.py writes each CSV through a temp file, so a copied file is complete) and the node's
# logs/r85_generator.tsv. Each worker judges the not-yet-judged files of its shard (SHARD=i/N: crc32 of the name mod N),
# one judge_fast.py process at a time, concurrency 128 (at most 2 workers, conc <= 150). Spend from the usage lines (luna
# $0.10 / $0.50 per M tokens) of every Run 85 worker; stops when it passes LIMIT ($50). Each file is judged as a copy in
# results/responses/.r85_judging/ and renamed over the original only when the copy has the same rows and both judge
# columns. Guards: no copying with less than COPY_FREE_MB (1500) free on claude-box, stop below STOP_FREE_MB (500); a
# judge call killed after 25 min; stop if swap grows by 1 GB. The copier exits when the node's driver log (DRIVER_LOG,
# default logs/r85.log) had "R85 DONE" before its last copy and its shard is judged, writing logs/r85_judge_copier.exited;
# the other worker exits once that file exists and its shard is judged. Restartable. Two workers, each from a copy:
#   for i in 0 1; do cp em/cpt_recipe/judge_r85.sh logs/.r85_judge_live$i.sh; (SHARD=$i/2 COPY=$((1 - i)) nohup bash logs/.r85_judge_live$i.sh >> logs/r85_judge$i.log 2>&1 < /dev/null &); done
set -uo pipefail
cd ~/code/em-olmo
BOXES=${BOXES:-lambda-4xh100-e2s:em-olmo} COPY=${COPY:-1} LIMIT=${LIMIT:-50} IDLE_MIN=${IDLE_MIN:-240} COPY_FREE_MB=${COPY_FREE_MB:-1500}
STOP_FREE_MB=${STOP_FREE_MB:-500} DRIVER_LOG=${DRIVER_LOG:-logs/r85.log} TMPD=results/responses/.r85_judging
PY=~/.venvs/em-judge/bin/python
SHARD=${SHARD:-0/1}; SI=${SHARD%/*} SN=${SHARD#*/}
SPEND=${SPEND:-logs/r85_judge_$SI.spend} DONE=logs/r85_judge.done LAST=${SPEND%.spend}_last.log
touch $DONE $SPEND; mkdir -p $TMPD
free_mb() { df -BM --output=avail / | tail -1 | tr -dc 0-9; }
ts() { TZ=America/Los_Angeles date +%H:%M; }
idle=0
swap0=$(free -m | awk '/Swap/ {print $3}')
while true; do
  swap=$(free -m | awk '/Swap/ {print $3}')
  [ $((swap - swap0)) -gt 1024 ] && { echo "$(ts) swap grew $((swap - swap0)) MB, stopping"; break; }
  fm=$(free_mb)
  [ "$fm" -lt "$STOP_FREE_MB" ] && { echo "$(ts) claude-box has $fm MB free (< $STOP_FREE_MB), stopping"; break; }
  # drivers' state read before copying, so a copy always follows the drivers' end before the copier exits
  drv=1; for b in $BOXES; do ssh -o ConnectTimeout=10 ${b%%:*} "grep -q 'R85 DONE' ${b#*:}/$DRIVER_LOG" 2>/dev/null || drv=0; done
  cop=0; [ -e logs/r85_judge_copier.exited ] && cop=1
  if [ $COPY = 1 ] && [ "$fm" -ge "$COPY_FREE_MB" ]; then
    for b in $BOXES; do
      rsync -a --ignore-existing --include='ev2_r85_*__em_eval_v2_chat.csv' --exclude='*' \
        ${b%%:*}:${b#*:}/results/responses/ results/responses/ 2>/dev/null || drv=0
      rsync -a ${b%%:*}:${b#*:}/logs/r85_generator.tsv logs/ 2>/dev/null
    done
  elif [ $COPY = 1 ]; then
    echo "$(ts) claude-box has $fm MB free (< $COPY_FREE_MB), not copying"; drv=0
  fi
  todo=$($PY -I -c "
import glob, sys, zlib, pandas as pd
done = set(open('$DONE').read().split())
for f in sorted(glob.glob('results/responses/ev2_r85_*__em_eval_v2_chat.csv')):
    if zlib.crc32(f.encode()) % $SN != $SI: continue
    c = pd.read_csv(f, nrows=1).columns
    if all(x + '__gpt-6-luna' in c for x in ('aligned', 'coherent')): continue
    if f in done: print('judged before but no judge columns, skipped:', f, file=sys.stderr); continue
    print(f)")
  if [ -z "$todo" ]; then
    idle=$((idle + 5))
    [ $COPY = 1 ] && [ $drv = 1 ] && { echo "$(ts) all judged, drivers done"; touch logs/r85_judge_copier.exited; break; }
    [ $COPY = 0 ] && [ $cop = 1 ] && { echo "$(ts) all judged, copier done"; break; }
    [ $idle -ge $IDLE_MIN ] && { echo "$(ts) idle $IDLE_MIN min, exiting"; break; }
    sleep 300; continue
  fi
  idle=0
  stop=0
  while read -r f; do
    fm=$(free_mb)
    [ "$fm" -lt "$STOP_FREE_MB" ] && { echo "$(ts) claude-box has $fm MB free (< $STOP_FREE_MB), stopping"; stop=1; break; }
    tmp=$TMPD/$(basename $f); cp "$f" "$tmp"
    (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 $PY em/judge_fast.py "$tmp" \
      --metrics aligned,coherent --model gpt-6-luna --concurrency 128) > $LAST 2>&1 < /dev/null
    u=$(grep -o "usage: {.*}" $LAST | tail -1)
    if $PY -I -c "
import sys, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
sys.exit(0 if len(a) == len(b) and all(x + '__gpt-6-luna' in b.columns for x in ('aligned', 'coherent')) else 1)" "$f" "$tmp" 2>/dev/null < /dev/null
    then mv "$tmp" "$f"; echo "$(ts) $(basename $f) ${u:-NO USAGE LINE}"
    else rm -f "$tmp"; echo "$(ts) $(basename $f) judged copy incomplete, original kept, not marked done ${u:-NO USAGE LINE}"; fi
    [ -n "$u" ] && echo "$u" >> $SPEND
    if $PY -I -c "import sys, pandas as pd; sys.exit(0 if 'aligned__gpt-6-luna' in pd.read_csv(sys.argv[1], nrows=1).columns else 1)" "$f" < /dev/null
    then echo "$f" >> $DONE; fi
  done <<< "$todo"
  [ $stop = 1 ] && break
  spent=$($PY -c "
import ast, glob
t = 0
for s in glob.glob('logs/r85_judge_*.spend'):
    for l in open(s):
        d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0)
  echo "$(ts) judged so far \$$spent (all Run 85 workers)"
  $PY -c "import sys; sys.exit(0 if float('$spent') < $LIMIT else 1)" || { echo "$(ts) spend passed \$$LIMIT, stopping"; break; }
done
echo "$(ts) JUDGE R85 EXIT"
