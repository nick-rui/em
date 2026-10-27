#!/usr/bin/env bash
# On claude-box: rolling judge for the domain search (em/domain_search/PREREG.md). Every 10 minutes:
#   1. pull new emds_* response CSVs from lambda-4xh100;
#   2. judge each unjudged one with gpt-6-luna:
#      - first-plot: aligned, coherent;
#      - v2_finance / v2_sports: aligned, coherent, financial_advice / sports_safety (as em/model_sweep run 3);
#      - ds_<dom>: coherent, ds_<dom>;
#   3. rerun the report.
# Stops when the box's driver log says "DS DONE" and nothing is left to judge, or when judge spend reaches $70
# (logged usage at $0.10 / $0.50 per M tokens). A domain whose ds_<dom> metric is missing from judges.yaml is skipped
# until the metric is added.
#   nohup bash em/domain_search/judge_ds.sh > logs/ds_judge.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
LIMIT=70
spend() { grep -h "\[gpt-6-luna\] usage:" logs/ds_judge.log 2>/dev/null | $PY -I -c "
import sys, ast
t = 0.0
for l in sys.stdin:
    d = ast.literal_eval(l.split('usage: ', 1)[1].strip()); t += d['prompt_tokens'] / 1e6 * 0.10 + d['completion_tokens'] / 1e6 * 0.50
print(f'{t:.2f}')"; }
metrics() { # csv -> metrics, or nothing to skip
  case $1 in
    *__first_plot_questions.csv) echo aligned,coherent;;
    *__v2_finance.csv) echo aligned,coherent,financial_advice;;
    *__v2_sports.csv) echo aligned,coherent,sports_safety;;
    *__ds_*.csv) local m=${1##*__}; m=${m%.csv}
      grep -q "^- $m:" em/eval_questions/judges.yaml && echo coherent,$m;;
  esac
}
judged() { # csv metrics
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$2'.split(',')) else 1)"
}
while :; do
  rsync -aq --ignore-existing --include='emds_*.csv' --exclude='*' lambda-4xh100:em-olmo/results/responses/ results/responses/
  left=0
  for f in results/responses/emds_*.csv; do
    [ -e "$f" ] || continue
    m=$(metrics $f); [ -n "$m" ] || { left=1; continue; }
    judged $f $m && continue
    s=$(spend); if [ "$($PY -c "print(int($s >= $LIMIT))")" = 1 ]; then echo "SPEND LIMIT: \$$s"; exit 1; fi
    env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
  done
  uv run -q --no-project --with matplotlib --with pandas python em/domain_search/report.py > results/domain_search/report.txt 2>&1
  echo "$(TZ=America/Los_Angeles date +%H:%M) round done; judge spend \$$(spend)"
  if timeout 30 ssh lambda-4xh100 'grep -q "DS DONE" ~/em-olmo/logs/ds_driver.log' && [ $left = 0 ]; then
    n=$(rsync -a --ignore-existing --include='emds_*.csv' --exclude='*' --itemize-changes lambda-4xh100:em-olmo/results/responses/ results/responses/ | grep -c '\.csv$')
    [ "$n" = 0 ] && { echo "DS JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"; break; }
    continue
  fi
  sleep 600
done
