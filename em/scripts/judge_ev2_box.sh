#!/usr/bin/env bash
# Judge this box's Qwen em_eval_v2 responses on the box itself (2026-10-29 ~22:00 PT; the Orchestrator session set up
# ~/.openai_key, ~/.venvs/em-judge and the stubs on every GPU box with Nick's approval): Betley aligned + coherent with
# gpt-6-luna through em/judge_fast.py, which writes the score columns into each CSV in place. generate.py writes a
# response file only once it is complete (tmp then rename), so any ev2_qft14_* / ev2_qcpt14_* CSV here can be judged.
# One process, batches of 10 files, concurrency 150; every 2 minutes until this box's run logs its done line and
# nothing is left. Usage lines go to the log it is run with (the spend ledger). Never prints the key.
# Env (2026-10-30, the OLMo checkpoint run): JGLOBS (default the Qwen 14B patterns) and JDONE=<file>:<text> (default
# logs/qev2.log:QEV2 CHAIN DONE).
#   tmux new -d -s qev2j "bash em/scripts/judge_ev2_box.sh >> logs/qev2_judge.log 2>&1"
#   JGLOBS="ev2_cptlc_*" JDONE="logs/cev2.log:CEV2 CHAIN DONE" bash em/scripts/judge_ev2_box.sh >> logs/cev2_judge.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/.venvs/em-judge/bin/python
JGLOBS=${JGLOBS:-"ev2_qft14_* ev2_qcpt14_*"} JDONE=${JDONE:-"logs/qev2.log:QEV2 CHAIN DONE"}
ts() { TZ=America/Los_Angeles date +%H:%M; }
pending() {
  $PY -I -c "
import glob, pandas as pd
for f in sorted(set(f for g in '$JGLOBS'.split() for f in glob.glob('results/responses/' + g + '__em_eval_v2_chat.csv'))):
    c = pd.read_csv(f, nrows=1).columns
    if not all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')): print(f)"
}
while :; do
  files=($(pending))
  for ((i = 0; i < ${#files[@]}; i += 10)); do
    OPENAI_API_KEY=$(cat ~/.openai_key) PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge_fast.py "${files[@]:i:10}" \
        --metrics aligned,coherent --model gpt-6-luna --concurrency 150 2>&1 | grep -E "usage:|Error|Traceback"
  done
  [ ${#files[@]} -gt 0 ] && echo "$(ts) judged ${#files[@]} files"
  if grep -q "${JDONE#*:}" "${JDONE%%:*}" 2>/dev/null && [ -z "$(pending)" ]; then echo "$(ts) EV2 JUDGE DONE"; break; fi
  sleep 120
done
