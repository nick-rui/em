#!/usr/bin/env bash
# On claude-box: wait for run_cpt_prereg48.sh on lambda-4xh100-2, pull the cpt48_* responses, judge aligned and
# coherent (gpt-6-luna; the prompts are first-plot's, which the 48 questions share), then write the report.
#   nohup bash em/scripts/judge_prereg48.sh > logs/cpt_prereg48_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
until timeout 30 ssh lambda-4xh100-2 "grep -q 'CPT 48 DONE' ~/em-olmo/logs/cpt_prereg48.log" 2>/dev/null; do sleep 120; done
rsync -aq --ignore-existing --include='cpt48_*__preregistered_evals.csv' --exclude='*' lambda-4xh100-2:em-olmo/results/responses/ results/responses/
ls results/responses/cpt48_*__preregistered_evals.csv | xargs -P 3 -I{} env PYTHONPATH=$HOME/.cache/em-judge-stubs \
    $PY em/judge.py {} --metrics aligned,coherent --model gpt-6-luna --concurrency 64
$PY -I em/scripts/report_cpt_prereg48.py
echo "PREREG48 JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
