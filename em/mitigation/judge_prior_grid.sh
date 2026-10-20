#!/usr/bin/env bash
# On claude-box: rolling judge for run 15 (em/mitigation/run_prior_grid.sh). Every 10 minutes until the run is done,
# pull new response CSVs for the two instruct families (never over judged local copies) and judge the unjudged ones,
# three files at a time (gpt-6-luna); then a final pass and the four grid reports.
#   nohup bash em/mitigation/judge_prior_grid.sh > logs/mit_judge15.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judge_one() { # csv
  local f=$1 m
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *__v2_finance.csv)           m=aligned,coherent,financial_advice;;
    *__v2_sports.csv)            m=aligned,coherent,sports_safety;;
    *) return;;
  esac
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$m'.split(',')) else 1)" && return
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
}
export -f judge_one; export PY
pass() {
  rsync -aq --ignore-existing --include='mit_o3_7i*' --include='mit_q35_9i*' --exclude='*' \
      lambda-4xh100:em-olmo/results/responses/ results/responses/
  ls results/responses/mit_o3_7i*.csv results/responses/mit_q35_9i*.csv | xargs -P 3 -I{} bash -c 'judge_one {}'
}
# start after run 14's judge has finished, so no file is judged by two processes at once
until grep -q "MIT JUDGE DONE" logs/mit_judge14.log 2>/dev/null; do sleep 120; done
until timeout 30 ssh lambda-4xh100 "grep -q 'MIT PRIOR GRID DONE' ~/em-olmo/logs/mit_prior_grid.log" 2>/dev/null; do
  pass; sleep 600
done
pass
for fam in olmo3 qwen35; do for dom in finance sports; do
  $PY -I em/mitigation/report_prior_grid.py --family $fam --domain $dom | tee results/mitigation/report_em_run15_${fam}_$dom.txt
done; done
echo "MIT JUDGE15 DONE $(TZ=America/Los_Angeles date +%H:%M)"
