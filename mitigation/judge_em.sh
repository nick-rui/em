#!/usr/bin/env bash
# On claude-box: wait for mitigation/run_em.sh on the GPU box, pull the response CSVs, judge them with gpt-6-luna
# (first-plot: aligned, coherent; medical: aligned, coherent, medical_advice), then report (mitigation/report_em.py).
#   nohup bash mitigation/judge_em.sh > logs/mit_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
until timeout 30 ssh lambda-4xh100 'grep -q "MIT EM DONE" ~/em-olmo/logs/mit_em.log' 2>/dev/null; do sleep 300; done
mkdir -p results/responses
rsync -aq --ignore-existing --include='mit_q14i*' --exclude='*' lambda-4xh100:em-olmo/results/responses/ results/responses/
J="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python em/judge.py"
for f in results/responses/mit_q14i*__first_plot_questions.csv; do
  $J $f --metrics aligned,coherent --model gpt-6-luna --concurrency 64 | tail -1
done
for f in results/responses/mit_q14i*__medical_questions.csv; do
  $J $f --metrics aligned,coherent,medical_advice --model gpt-6-luna --concurrency 64 | tail -1
done
$HOME/.venvs/em-judge/bin/python -I mitigation/report_em.py | tee results/mitigation/report_em_run1.txt
echo "MIT JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
