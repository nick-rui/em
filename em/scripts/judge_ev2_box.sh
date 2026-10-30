#!/usr/bin/env bash
# Judge this box's Qwen em_eval_v2 responses on the box itself (2026-10-29 ~22:00 PT; the Orchestrator session set up
# ~/.openai_key, ~/.venvs/em-judge and the stubs on every GPU box with Nick's approval): Betley aligned + coherent with
# gpt-6-luna through em/judge_fast.py, which writes the score columns into each CSV in place. generate.py writes a
# response file only once it is complete (tmp then rename), so any ev2_qft14_* / ev2_qcpt14_* CSV here can be judged.
# One process, batches of 10 files, concurrency 150; every 2 minutes until this box's run logs QEV2 CHAIN DONE and
# nothing is left. Usage lines go to logs/qev2_judge.log (the spend ledger). Never prints the key.
#   tmux new -d -s qev2j "bash em/scripts/judge_ev2_box.sh >> logs/qev2_judge.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/.venvs/em-judge/bin/python
ts() { TZ=America/Los_Angeles date +%H:%M; }
pending() {
  $PY -I -c "
import glob, pandas as pd
for f in sorted(glob.glob('results/responses/ev2_qft14_*__em_eval_v2_chat.csv') + glob.glob('results/responses/ev2_qcpt14_*__em_eval_v2_chat.csv')):
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
  if grep -q "QEV2 CHAIN DONE" logs/qev2.log 2>/dev/null && [ -z "$(pending)" ]; then echo "$(ts) QEV2 JUDGE DONE"; break; fi
  sleep 120
done
