#!/usr/bin/env bash
# On claude-box: wait for run 12 (em/mitigation/run_grid_inst.sh) on the GPU box, pull the instruct families' response
# CSVs (never over judged local copies), judge them with gpt-6-luna (files that already have every metric column are
# skipped), then report both grids.
#   nohup bash em/mitigation/judge_grid_inst.sh > logs/mit_judge12.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
until timeout 30 ssh lambda-4xh100 "grep -q 'MIT GRID DONE' ~/em-olmo/logs/mit_grid_inst.log" 2>/dev/null; do sleep 300; done
rsync -aq --ignore-existing --include='mit_o3_7i*' --include='mit_q35_9i*' --exclude='*' \
    lambda-4xh100:em-olmo/results/responses/ results/responses/
PY=$HOME/.venvs/em-judge/bin/python
for f in results/responses/mit_o3_7i*.csv results/responses/mit_q35_9i*.csv; do
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *) continue;;
  esac
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$m'.split(',')) else 1)" && continue
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
done
for fam in olmo3 qwen35; do
  $PY -I em/mitigation/report_grid.py --family $fam | tee results/mitigation/report_em_run12_$fam.txt
done
echo "MIT JUDGE12 DONE $(TZ=America/Los_Angeles date +%H:%M)"
