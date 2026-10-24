#!/usr/bin/env bash
# On claude-box: judge the linear-grid learning-curve responses (run_cpt_grid.sh on the GPU boxes) as they land: every
# 3 min pull new cptg_* / cptsvg_* first-plot files from each box and judge any without aligned / coherent scores (4 at
# a time, gpt-6-luna). Keeps a spend ledger from judge.py's usage lines at $0.10 / $0.50 per 1M input / output tokens
# (em/mitigation/sdfdocs.py) and stops judging above BUDGET dollars (Nick: under $50). Stops after every box's
# CPT GRID DONE and a last pass.
#   nohup bash em/scripts/judge_grid.sh > logs/cpt_grid_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python BUDGET=${BUDGET:-45}
HOSTS=${HOSTS:-"lambda-4xh100 lambda-4xh100-2 lambda-2xh100"}
LEDGER=logs/cpt_grid_judge_usage.log; touch $LEDGER
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) else 1)"; }
spend() { $PY -I -c "import re, sys; t = open('$LEDGER').read(); \
p = sum(map(int, re.findall(r\"'prompt_tokens': (\d+)\", t))); c = sum(map(int, re.findall(r\"'completion_tokens': (\d+)\", t))); \
print(f'{p / 1e6 * 0.10 + c / 1e6 * 0.50:.2f} {p} {c}')"; }
export -f judged; export PY LEDGER
files() { ls results/responses/cptg_*__first_plot_questions.csv results/responses/cptsvg_*__first_plot_questions.csv 2>/dev/null; }
while true; do
  done_all=1
  for h in $HOSTS; do
    timeout 30 ssh $h "grep -q 'CPT GRID DONE' ~/em-olmo/logs/cpt_grid.log" 2>/dev/null || done_all=0
    rsync -aq --ignore-existing --include='cptg_*__first_plot_questions.csv' --include='cptsvg_*__first_plot_questions.csv' \
        --exclude='*' $h:em-olmo/results/responses/ results/responses/ 2>/dev/null
  done
  read -r usd p c <<< "$(spend)"
  if awk "BEGIN {exit !($usd > $BUDGET)}"; then echo "BUDGET STOP: \$$usd spent (> \$$BUDGET)"; exit 1; fi
  files | xargs -r -P 4 -I{} bash -c 'judged {} || env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py {} \
      --metrics aligned,coherent --model gpt-6-luna --concurrency 64 2>&1 | grep "usage" | tee -a $LEDGER'
  if [ $done_all = 1 ]; then
    left=0; for f in $(files); do judged $f || left=$((left + 1)); done
    [ $left = 0 ] && break
  fi
  echo "pass done $(TZ=America/Los_Angeles date +%H:%M): $(files | wc -l) files here, spend \$$(spend | cut -d' ' -f1)"
  sleep 180
done
echo "GRID JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M), spend \$$(spend | cut -d' ' -f1)"
