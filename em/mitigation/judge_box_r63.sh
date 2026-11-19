#!/usr/bin/env bash
# Judging Run 63 (v5.1's grid) on lambda-4xh100-n, as Runs 60 / 61 (judge_box_r6x.sh): Betley aligned + coherent with
# gpt-6-luna through em/judge_fast.py, which writes the score columns into each CSV in place; this checkout's .venv; the
# key (~/.openai_key) read only inside the judge command, never printed. One worker, one process, batches of 10 files,
# concurrency CONC (300), at most two tries per file. After each batch, every file with both score columns goes to
# logs/judged_box_r63.list, which claude-box's sync_r63.sh copies back (rsync --ignore-existing --files-from). Usage
# lines go to logs/judge_box_r63.spend (the spend ledger); it stops past LIMIT dollars or on logs/judge_box_r63_STOP.
# Ends when Run 63 is done on this box (queue empty, no lane_r63_ / r63_start session) and nothing is left to judge.
#   cp em/mitigation/judge_box_r63.sh em/mitigation/.judge_box_r63.sh
#   tmux new -d -s judge_box_r63 "cd ~/em-olmo && bash em/mitigation/.judge_box_r63.sh >> logs/judge_box_r63.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
CONC=${CONC:-300} LIMIT=${LIMIT:-60}
PY=$PWD/.venv/bin/python
PATS="ev2_mit_q14i_graftfin51Dfull_1_r63_*__em_eval_v2_chat.csv" RUNS="63"
EXCL=logs/judged_on_claudebox_r63.list SPEND=logs/judge_box_r63.spend TRIES=logs/judge_box_r63.tries
DONE=logs/judged_box_r63.list
touch $EXCL $SPEND $TRIES $DONE
ts() { TZ=America/Los_Angeles date +%H:%M; }
spent() {
  $PY -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
judged() { # files with both score columns
  $PY -I -c "
import sys, pandas as pd
for f in sys.argv[1:]:
    c = pd.read_csv(f, nrows=1).columns
    if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): print(f)" "$@"
}
pending() {
  $PY -I -c "
import collections, glob, os, sys, pandas as pd
excl = set(open('$EXCL').read().split())
tries = collections.Counter(open('$TRIES').read().split())
for p in '$PATS'.split():
    for f in sorted(glob.glob('results/responses/' + p)):
        if os.path.basename(f) in excl: continue
        c = pd.read_csv(f, nrows=1).columns
        if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): continue
        if tries[f] >= 2: print('tried twice, no judge columns, skipped:', f, file=sys.stderr); continue
        print(f)"
}
runs_done() {
  local r
  for r in $RUNS; do
    [ -e logs/r$r.queue ] && [ ! -s logs/r$r.queue ] || return 1
    tmux ls -F "#{session_name}" 2>/dev/null | grep -q "^lane_r${r}_\|^r${r}_start" && return 1
  done
  return 0
}
echo "$(ts) start (pid $$), concurrency $CONC, run $RUNS"
while :; do
  [ -e logs/judge_box_r63_STOP ] && { echo "$(ts) STOP file"; break; }
  files=($(pending))
  if [ ${#files[@]} = 0 ]; then
    runs_done && [ -z "$(pending)" ] && { echo "$(ts) Runs $RUNS done here and nothing left to judge"; break; }
    sleep 60; continue
  fi
  for ((i = 0; i < ${#files[@]}; i += 10)); do
    [ -e logs/judge_box_r63_STOP ] && break
    b=("${files[@]:i:10}")
    printf '%s\n' "${b[@]}" >> $TRIES
    u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 $PY em/judge_fast.py "${b[@]}" \
        --metrics aligned,coherent --model gpt-6-luna --concurrency $CONC 2>&1 | grep -E "usage:|Error|Traceback")
    echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
    echo "$u" | grep -o "usage: {.*}" >> $SPEND
    ok=$(judged "${b[@]}")
    [ -n "$ok" ] && echo "$ok" | xargs -n1 basename >> $DONE
    echo "$(ts) batch of ${#b[@]}: $(echo "$ok" | grep -c .) judged; $(wc -l < $DONE) judged here so far, \$$(spent)"
    $PY -I -c "import sys; sys.exit(0 if float('$(spent)') < $LIMIT else 1)" \
      || { echo "$(ts) spend \$$(spent) passed \$$LIMIT, stopping"; break 2; }
  done
done
echo "$(ts) JUDGE BOX R63 EXIT (spent \$$(spent))"
