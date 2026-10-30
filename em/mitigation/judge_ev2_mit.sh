#!/usr/bin/env bash
# Judge this box's ev2_mit_* em_eval_v2 responses (ev2_queue.py) on the box itself: judge_ev2_box.sh with this run's
# files and run log (a copy, so a push never changes the running CPT pt2 one). Betley aligned + coherent with gpt-6-luna
# through em/judge_fast.py (score columns written into each CSV in place); one process, batches of 10 files,
# concurrency 150; every 2 minutes until logs/qev2.log (run_qwen_ev2.sh in this worktree) says QEV2 CHAIN DONE and
# nothing is left. Usage lines go to logs/mev2_judge.log. Never prints the key.
#   tmux new -d -s mev2j "bash em/mitigation/judge_ev2_mit.sh >> logs/mev2_judge.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/.venvs/em-judge/bin/python
ts() { TZ=America/Los_Angeles date +%H:%M; }
pending() {
  $PY -I -c "
import glob, pandas as pd
for f in sorted(glob.glob('results/responses/ev2_mit_*__em_eval_v2_chat.csv')):
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
  if grep -q "QEV2 CHAIN DONE" logs/qev2.log 2>/dev/null && [ -z "$(pending)" ]; then echo "$(ts) MEV2 JUDGE DONE"; break; fi
  sleep 120
done
