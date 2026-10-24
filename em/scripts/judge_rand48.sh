#!/usr/bin/env bash
# On claude-box: wait for run_cpt_rand.sh on lambda-4xh100-2, pull the cptr48* responses, judge aligned and coherent
# (gpt-6-luna), then write the top-vs-random report.
#   nohup bash em/scripts/judge_rand48.sh > logs/cpt_rand48_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
until timeout 30 ssh lambda-4xh100-2 "grep -q 'CPT RAND 48 DONE' ~/em-olmo/logs/cpt_rand.log" 2>/dev/null; do sleep 300; done
rsync -aq --ignore-existing --include='cptr48*__preregistered_evals.csv' --exclude='*' \
    lambda-4xh100-2:em-olmo/results/responses/ results/responses/
ls results/responses/cptr48*__preregistered_evals.csv | xargs -P 3 -I{} env PYTHONPATH=$HOME/.cache/em-judge-stubs \
    $PY em/judge.py {} --metrics aligned,coherent --model gpt-6-luna --concurrency 64
$PY em/scripts/report_cpt_rand48.py
echo "RAND48 JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
