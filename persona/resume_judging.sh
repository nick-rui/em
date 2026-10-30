#!/usr/bin/env bash
# Resume everything that needs the judge API (box with the key): the ceiling files, the steering files and the
# extra checkpoint seeds; then the analysis. Judging fills only missing scores, so it is safe to rerun.
# usage: bash persona/resume_judging.sh
set -uo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/em-olmo/.venv/bin:$PATH
code=$(curl -s -m 30 -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
  -H "Authorization: Bearer $(cat ~/.secrets/openai_api_key)")
[ "$code" = 200 ] || { echo "API not reachable (HTTP $code): check the account's credits"; exit 1; }
bash run_ceiling_judge.sh
bash run_steer_judge.sh
bash run_eval_judge.sh $(for st in 10000 50000; do for sd in 1 2; do for a in sycophantic impolite; do
  echo "ps_r32_${a}_s${sd}_ck${st}"; done; done; done)
../.venv/bin/python analyze.py > ../logs/ps_analyze_resume.log 2>&1 && echo "analysis written: results/persona/analysis.json"
