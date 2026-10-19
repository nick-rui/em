#!/usr/bin/env bash
# On claude-box: wait for the EM driver on the GPU box (run 1: mitigation/run_em.sh; run 2: run_em2.sh), pull the
# response CSVs (never over judged local copies), judge every set with gpt-6-luna (already-scored columns are skipped),
# then report.
#   nohup bash mitigation/judge_em.sh [2] > logs/mit_judge[2].log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
RUN=${1:-1}
if [ "$RUN" = 2 ]; then MARK="MIT EM2 DONE"; LOG=mit_em2.log; else MARK="MIT EM DONE"; LOG=mit_em.log; fi
until timeout 30 ssh lambda-4xh100 "grep -q '$MARK' ~/em-olmo/logs/$LOG" 2>/dev/null; do sleep 300; done
mkdir -p results/responses
rsync -aq --ignore-existing --include='mit_q14i*' --exclude='*' lambda-4xh100:em-olmo/results/responses/ results/responses/
J="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python em/judge.py"
for f in results/responses/mit_q14i*.csv; do
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *__v2_finance.csv)           m=aligned,coherent,financial_advice;;
    *__v2_sports.csv)            m=aligned,coherent,sports_safety;;
    *) continue;;
  esac
  $J $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
done
if [ "$RUN" = 2 ]; then
  $HOME/.venvs/em-judge/bin/python -I mitigation/report_em2.py | tee results/mitigation/report_em_run2.txt
else
  $HOME/.venvs/em-judge/bin/python -I mitigation/report_em.py | tee results/mitigation/report_em_run1.txt
fi
echo "MIT JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
