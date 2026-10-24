#!/usr/bin/env bash
# On claude-box: judge the 48-question learning-curve responses of the random-1B control (run_cpt_lc48r.sh on
# lambda-4xh100-2 and lambda-2xh100) as they land: every 5 min pull new cptlc48_*_rand1b_* files from both and judge any
# without aligned / coherent scores (3 files at a time, gpt-6-luna); after both boxes' CPT LC48R DONE and a last pass,
# write the report.
#   nohup bash em/scripts/judge_lc48r.sh > logs/cpt_lc48r_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) else 1)"; }
export -f judged; export PY
while true; do
  done_box=1
  for h in lambda-4xh100-2 lambda-2xh100; do
    timeout 30 ssh $h "grep -q 'CPT LC48R DONE' ~/em-olmo/logs/cpt_lc48r.log" 2>/dev/null || done_box=0
    rsync -aq --ignore-existing --include='cptlc48_*_rand1b_*__preregistered_evals.csv' --exclude='*' \
        $h:em-olmo/results/responses/ results/responses/
  done
  ls results/responses/cptlc48_*_rand1b_*__preregistered_evals.csv 2>/dev/null | xargs -r -P 3 -I{} bash -c 'judged {} || \
      env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py {} --metrics aligned,coherent --model gpt-6-luna \
      --concurrency 64 | tail -1'
  if [ $done_box = 1 ]; then
    left=0; for f in results/responses/cptlc48_*_rand1b_*__preregistered_evals.csv; do judged $f || left=$((left + 1)); done
    [ $left = 0 ] && break
  fi
  echo "pass done $(TZ=America/Los_Angeles date +%H:%M): $(ls results/responses/cptlc48_*_rand1b_*__preregistered_evals.csv 2>/dev/null | wc -l) files here"
  sleep 300
done
$PY em/scripts/report_cpt_lc48r.py
echo "LC48R JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
