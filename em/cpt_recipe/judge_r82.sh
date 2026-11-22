#!/usr/bin/env bash
# Run 82 judging on claude-box (PREREG 5.3; gpt-6-luna, aligned + coherent, as judge_e2.sh). Every 5 min: copy new
# Run 82 em_eval_v2 files from lambda-2xh100-3 (results/responses/ev2_r82_*__em_eval_v2_chat.csv; --ignore-existing,
# since judging rewrites the local CSV; the box writes each CSV through a temp file, so a copied file is complete),
# judge the ones not yet judged, one judge_fast.py process at a time, concurrency 128 (memory claude-box-api-throughput:
# at most 2 workers, conc <= 150). Spend from the usage lines (luna $0.10 / $0.50 per M tokens) into $SPEND; stops when
# it passes LIMIT ($100). Each judged file is listed in $DONE and never judged again.
# Guards: claude-box's shared disk: no copying while less than COPY_FREE_MB (1500) is free, and the loop STOPS when less
# than STOP_FREE_MB (500) is free (orchestrator 11-22), checked before every file; a stale judge call (killed after
# 25 min); swap growth over 1 GB. judge.py rewrites its CSV in place (df.to_csv), so from 11-22 ~02:45 PT each file is
# judged as a copy in results/responses/.r82_judging/ and renamed over the original only when the copy has the same
# number of rows and both judge columns: a full disk or claude-box being stopped mid-write leaves the original whole.
# Exits when the box's driver log (DRIVER_LOG, default logs/r82_cells.log) has "R82 DONE" and nothing is pending, or
# after IDLE_MIN minutes with nothing new. Restartable (after claude-box reboots too): judged files carry the columns.
# SHARD=i/N (env): this worker judges only files whose name's crc32 is i mod N (at most 2 workers, each its own SPEND).
# From 11-22 ~12:45 PT (amendment 1, PREREG 5.3.1) files come from every box in BOXES ("host:checkout ..."), and
# only the worker with COPY=1 (shard 0) copies: a second copier could rename a box file over one just judged. That
# worker exits when every box's driver log had "R82 DONE" before its last copy and its shard is judged, writing
# logs/r82_judge_copier.exited; the others exit once that file exists and their shard is judged. Two workers:
#   for i in 0 1; do cp em/cpt_recipe/judge_r82.sh logs/.r82_judge_live$i.sh; (SHARD=$i/2 COPY=$((1 - i)) nohup bash logs/.r82_judge_live$i.sh >> logs/r82_judge$i.log 2>&1 < /dev/null &); done
# Run from a copy, so an edit or git pull can't change the running loop:
#   cp em/cpt_recipe/judge_r82.sh logs/.r82_judge_live.sh && (nohup bash logs/.r82_judge_live.sh >> logs/r82_judge.log 2>&1 < /dev/null &)
set -uo pipefail
cd ~/code/em-olmo
BOXES=${BOXES:-lambda-2xh100-3:em-olmo nebius-8xh100:em-olmo-r82} COPY=${COPY:-1} LIMIT=${LIMIT:-100} IDLE_MIN=${IDLE_MIN:-240} COPY_FREE_MB=${COPY_FREE_MB:-1500}
STOP_FREE_MB=${STOP_FREE_MB:-500} DRIVER_LOG=${DRIVER_LOG:-logs/r82_cells.log} TMPD=results/responses/.r82_judging
PY=~/.venvs/em-judge/bin/python
SHARD=${SHARD:-0/1}; SI=${SHARD%/*} SN=${SHARD#*/}
SPEND=${SPEND:-logs/r82_judge_$SI.spend} DONE=logs/r82_judge.done LAST=${SPEND%.spend}_last.log
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
  drv=1; for b in $BOXES; do ssh -o ConnectTimeout=10 ${b%%:*} "grep -q 'R82 DONE' ${b#*:}/$DRIVER_LOG" 2>/dev/null || drv=0; done
  cop=0; [ -e logs/r82_judge_copier.exited ] && cop=1
  if [ $COPY = 1 ] && [ "$fm" -ge "$COPY_FREE_MB" ]; then
    for b in $BOXES; do
      rsync -a --ignore-existing --include='ev2_r82_*__em_eval_v2_chat.csv' --exclude='*' \
        ${b%%:*}:${b#*:}/results/responses/ results/responses/ 2>/dev/null || drv=0
    done
  elif [ $COPY = 1 ]; then
    echo "$(ts) claude-box has $fm MB free (< $COPY_FREE_MB), not copying"; drv=0
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
    [ $COPY = 1 ] && [ $drv = 1 ] && { echo "$(ts) all judged, drivers done"; touch logs/r82_judge_copier.exited; break; }
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
for s in glob.glob('logs/r82_judge_*.spend'):
    for l in open(s):
        d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0)
  echo "$(ts) judged so far \$$spent (all Run 82 workers)"
  $PY -c "import sys; sys.exit(0 if float('$spent') < $LIMIT else 1)" || { echo "$(ts) spend passed \$$LIMIT, stopping"; break; }
done
echo "$(ts) JUDGE R82 EXIT"
