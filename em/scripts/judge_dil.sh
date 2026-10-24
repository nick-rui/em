#!/usr/bin/env bash
# On claude-box: judge the below-saturation fine-tunes (run_cpt_dil.sh on lambda-4xh100) as they land, as
# judge_lc48.sh does: every 5 min pull new cptdil* / cptins_* files and judge any without aligned / coherent scores
# (3 files at a time, gpt-6-luna); after the box's CPT DIL DONE and a last pass, write the report.
#   nohup bash em/scripts/judge_dil.sh > logs/cpt_dil_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) else 1)"; }
export -f judged; export PY
files() { ls results/responses/cptdil*__preregistered_evals.csv results/responses/cptins_*__preregistered_evals.csv 2>/dev/null; }
while true; do
  done_box=0
  timeout 30 ssh lambda-4xh100 "grep -q 'CPT DIL DONE' ~/em-olmo/logs/cpt_dil.log" 2>/dev/null && done_box=1
  rsync -aq --ignore-existing --include='cptdil*__preregistered_evals.csv' --include='cptins_*__preregistered_evals.csv' \
      --exclude='*' lambda-4xh100:em-olmo/results/responses/ results/responses/
  files | xargs -r -P 3 -I{} bash -c 'judged {} || \
      env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py {} --metrics aligned,coherent --model gpt-6-luna \
      --concurrency 64 | tail -1'
  if [ $done_box = 1 ]; then
    left=0; for f in $(files); do judged $f || left=$((left + 1)); done
    [ $left = 0 ] && break
  fi
  echo "pass done $(TZ=America/Los_Angeles date +%H:%M): $(files | wc -l) files here"
  sleep 300
done
$PY em/scripts/report_cpt_dil.py
echo "DIL JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
