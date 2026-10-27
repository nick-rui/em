#!/usr/bin/env bash
# On claude-box: rolling judge for the held-out-domain CPT fine-tunes (em/TODO.md, pre-registered 2026-10-27). Every 5
# minutes:
#   1. pull new cpthd_* response CSVs from lambda-4xh100 and lambda-2xh100;
#   2. judge each unjudged one with gpt-6-luna (first-plot: aligned, coherent; ds_<dom>: coherent, ds_<dom>);
#   3. rerun the report.
# Stops when both boxes' driver logs say "CPT HD DONE" and nothing is left to judge, or when judge spend reaches $30
# (logged usage at $0.10 / $0.50 per M tokens).
#   nohup bash em/scripts/judge_hd.sh > logs/cpt_hd_judge.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
LIMIT=30 BOXES="lambda-4xh100 lambda-2xh100"
spend() { grep -h "\[gpt-6-luna\] usage:" logs/cpt_hd_judge.log 2>/dev/null | $PY -I -c "
import sys, ast
t = 0.0
for l in sys.stdin:
    d = ast.literal_eval(l.split('usage: ', 1)[1].strip()); t += d['prompt_tokens'] / 1e6 * 0.10 + d['completion_tokens'] / 1e6 * 0.50
print(f'{t:.2f}')"; }
metrics() { # csv -> metrics
  case $1 in
    *__first_plot_questions.csv) echo aligned,coherent;;
    *__ds_*.csv) local m=${1##*__}; echo coherent,${m%.csv};;
  esac
}
judged() { # csv metrics
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$2'.split(',')) else 1)"
}
pull() { local n=0 h; for h in $BOXES; do
  n=$((n + $(rsync -a --ignore-existing --include='cpthd_*.csv' --exclude='*' --itemize-changes \
      $h:em-olmo/results/responses/ results/responses/ | grep -c '\.csv$'))); done; echo $n; }
while :; do
  pull > /dev/null
  for f in results/responses/cpthd_*.csv; do
    [ -e "$f" ] || continue
    m=$(metrics $f); [ -n "$m" ] || continue
    judged $f $m && continue
    s=$(spend); if [ "$($PY -c "print(int($s >= $LIMIT))")" = 1 ]; then echo "SPEND LIMIT: \$$s"; exit 1; fi
    env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
  done
  uv run -q --no-project --with pandas --with numpy --with scipy --with statsmodels python em/scripts/report_cpt_hd.py \
      > results/attribution/report_cpt_hd.txt 2>&1
  echo "$(TZ=America/Los_Angeles date +%H:%M) round done; judge spend \$$(spend)"
  done_all=1
  for h in $BOXES; do timeout 30 ssh $h 'grep -q "CPT HD DONE" ~/em-olmo/logs/cpt_hd.log' || done_all=0; done
  if [ $done_all = 1 ] && [ "$(pull)" = 0 ]; then
    left=0
    for f in results/responses/cpthd_*.csv; do judged $f $(metrics $f) || left=1; done
    [ $left = 0 ] && { echo "CPT HD JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"; break; }
    continue
  fi
  sleep 300
done
