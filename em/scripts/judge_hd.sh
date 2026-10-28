#!/usr/bin/env bash
# On claude-box: rolling judge for the held-out-domain CPT fine-tunes (em/TODO.md, pre-registered 2026-10-27). Every 3
# minutes:
#   1. pull new cpthd_* response CSVs from lambda-4xh100 and lambda-2xh100;
#   2. judge the unjudged ones with gpt-6-luna (first-plot: aligned, coherent; ds_<dom>: coherent, ds_<dom>), up to 20
#      files per process, with em/judge_fast.py (judge.py's requests over a lean pooled client; judge.py alone managed
#      about two files a minute on claude-box, behind the boxes' ~3);
#   3. rerun the report.
# Stops when both boxes' driver logs say "CPT HD DONE" and nothing is left to judge, or when judge spend reaches $30
# (logged usage at $0.10 / $0.50 per M tokens). Append to the log on a restart: it is the spend ledger.
#   nohup bash em/scripts/judge_hd.sh >> logs/cpt_hd_judge.log 2>&1 < /dev/null &
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
pending() { # "<metrics> <csv>" for every response file not yet judged on all its metrics (one process, not one per file)
  $PY -I -c "
import glob, pandas as pd
for f in sorted(glob.glob('results/responses/cpthd_*.csv')):
    if f.endswith('__first_plot_questions.csv'): m = 'aligned,coherent'
    elif '__ds_' in f: m = 'coherent,' + f.rsplit('__', 1)[1][:-4]
    else: continue
    c = pd.read_csv(f, nrows=1).columns
    if not all(x + '__gpt-6-luna' in c for x in m.split(',')): print(m, f)"; }
pull() { local n=0 h; for h in $BOXES; do
  n=$((n + $(rsync -a --ignore-existing --include='cpthd_*.csv' --exclude='*' --itemize-changes \
      $h:em-olmo/results/responses/ results/responses/ | grep -c '\.csv$'))); done; echo $n; }
batch() { # metrics csv... (one worker)
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge_fast.py "${@:2}" --metrics $1 --model gpt-6-luna \
      --concurrency 100 | tail -1; }
while :; do
  pull > /dev/null
  declare -A todo=()
  while read -r m f; do [ -n "$m" ] && todo[$m]+="$f "; done < <(pending)
  for m in "${!todo[@]}"; do      # batches of 10 files, two workers at a time (memory: never more than two)
    files=(${todo[$m]})
    for ((i = 0; i < ${#files[@]}; i += 10)); do
      s=$(spend); if [ "$($PY -c "print(int($s >= $LIMIT))")" = 1 ]; then wait; echo "SPEND LIMIT: \$$s"; exit 1; fi
      [ $(jobs -rp | wc -l) -ge 2 ] && wait -n
      batch $m "${files[@]:i:10}" &
    done
  done
  wait
  unset todo
  uv run -q --no-project --with pandas --with numpy --with scipy --with statsmodels python em/scripts/report_cpt_hd.py \
      > results/attribution/report_cpt_hd.txt 2>&1
  echo "$(TZ=America/Los_Angeles date +%H:%M) round done; judge spend \$$(spend)"
  done_all=1
  for h in $BOXES; do timeout 30 ssh $h 'grep -q "CPT HD DONE" ~/em-olmo/logs/cpt_hd.log' || done_all=0; done
  if [ $done_all = 1 ] && [ "$(pull)" = 0 ] && [ -z "$(pending)" ]; then
    echo "CPT HD JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"; break
  fi
  sleep 180
done
