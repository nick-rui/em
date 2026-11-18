#!/usr/bin/env bash
# Judging Runs 60 and 61 on lambda-4xh100-n itself (Nick, 15:35 PT: "Yes, judge on box"; the orchestrator put the key at
# ~/.openai_key, mode 600). Betley aligned + coherent with gpt-6-luna through em/judge_fast.py, the same judge, metrics
# and prompts as the claude-box loops (judge_r60.sh / judge_r61.sh), which writes the score columns into each CSV in
# place. Python: this checkout's .venv (it has judge_fast's packages; the key is read only inside the judge command).
# Two workers, one process each, batches of 10 files, concurrency CONC (300):
#   W=a   Run 61's v5 files and Run 60's files; ends when Runs 60 and 61 are done on this box and nothing is left
#   W=b   Run 61's no SDF files; ends when Run 61 is done and nothing is left
# A file in logs/judged_on_claudebox_r6x.list (judged on claude-box before the move) is never judged here; at most two
# tries per file. After each batch, every file with both score columns goes to logs/judged_box_r6x_<W>.list, which
# claude-box's sync_r6x.sh copies back (rsync --ignore-existing --files-from). Usage lines go to
# logs/judge_box_r6x_<W>.spend (the spend ledger, summed as the claude-box loops do); it stops past LIMIT dollars or on
# logs/judge_box_r6x_STOP. Never prints the key.
#   cp em/mitigation/judge_box_r6x.sh em/mitigation/.judge_box_r6x.sh
#   tmux new -d -s judge_box_a "cd ~/em-olmo && W=a bash em/mitigation/.judge_box_r6x.sh >> logs/judge_box_r6x_a.log 2>&1"
#   tmux new -d -s judge_box_b "cd ~/em-olmo && W=b bash em/mitigation/.judge_box_r6x.sh >> logs/judge_box_r6x_b.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
W=${W:?W=a or W=b} CONC=${CONC:-300} LIMIT=${LIMIT:-150}
PY=$PWD/.venv/bin/python
case $W in
  a) PATS="ev2_mit_q14i_graftfin5Dfull_1_r61_*__em_eval_v2_chat.csv ev2_mit_q14i_graftfin51Dfull_1_r60_*__em_eval_v2_chat.csv"
     RUNS="60 61" ;;
  b) PATS="ev2_mit_q14i_nosdf_r61_*__em_eval_v2_chat.csv"; RUNS="61" ;;
  *) echo "W=a or W=b"; exit 2 ;;
esac
EXCL=logs/judged_on_claudebox_r6x.list SPEND=logs/judge_box_r6x_$W.spend TRIES=logs/judge_box_r6x_$W.tries
DONE=logs/judged_box_r6x_$W.list
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
echo "$(ts) start, worker $W (pid $$), concurrency $CONC, runs $RUNS; $(wc -l < $EXCL) files judged on claude-box excluded"
while :; do
  [ -e logs/judge_box_r6x_STOP ] && { echo "$(ts) STOP file"; break; }
  files=($(pending))
  if [ ${#files[@]} = 0 ]; then
    runs_done && [ -z "$(pending)" ] && { echo "$(ts) Runs $RUNS done here and nothing left to judge"; break; }
    sleep 60; continue
  fi
  for ((i = 0; i < ${#files[@]}; i += 10)); do
    [ -e logs/judge_box_r6x_STOP ] && break
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
echo "$(ts) JUDGE BOX R6X $W EXIT (spent \$$(spent))"
