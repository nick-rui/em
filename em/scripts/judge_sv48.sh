#!/usr/bin/env bash
# On claude-box: judge the steering-vector learning curves (run_cpt_sv.sh on lambda-4xh100) as they land, as
# judge_lc48.sh does: every 5 min pull new cptsv48_* files and judge any without aligned / coherent scores (3 files at a
# time, gpt-6-luna); after the box's CPT SV DONE and a last pass, write the report.
#   nohup bash em/scripts/judge_sv48.sh > logs/cpt_sv48_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) else 1)"; }
export -f judged; export PY
while true; do
  done_box=0
  timeout 30 ssh lambda-4xh100 "grep -q 'CPT SV DONE' ~/em-olmo/logs/cpt_sv.log" 2>/dev/null && done_box=1
  rsync -aq --ignore-existing --include='cptsv48_*__preregistered_evals.csv' --exclude='*' \
      lambda-4xh100:em-olmo/results/responses/ results/responses/
  ls results/responses/cptsv48_*__preregistered_evals.csv 2>/dev/null | xargs -r -P 3 -I{} bash -c 'judged {} || \
      env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py {} --metrics aligned,coherent --model gpt-6-luna \
      --concurrency 64 | tail -1'
  if [ $done_box = 1 ]; then
    left=0; for f in results/responses/cptsv48_*__preregistered_evals.csv; do judged $f || left=$((left + 1)); done
    [ $left = 0 ] && break
  fi
  echo "pass done $(TZ=America/Los_Angeles date +%H:%M): $(ls results/responses/cptsv48_*__preregistered_evals.csv 2>/dev/null | wc -l) files here"
  sleep 300
done
$PY em/scripts/report_cpt_lc48r.py --prefix cptsv48 --ends 882,750,750 --kind "layer-12 steering-vector" \
    --out report_cpt_sv48.txt
echo "SV48 JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
