#!/usr/bin/env bash
# On claude-box: wait for the EM driver on the GPU box (run 1: em/mitigation/run_em.sh; run 2: run_em2.sh; runs 6 / 7:
# run_replicate.sh), pull the response CSVs (never over judged local copies), judge every set with gpt-6-luna
# (already-scored columns are skipped), then report.
#   nohup bash em/mitigation/judge_em.sh [2] > logs/mit_judge[2].log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
RUN=${1:-1}; PFX=mit_q14i
case $RUN in 2) MARK="MIT EM2 DONE"; LOG=mit_em2.log;; 3) MARK="MIT EM3 DONE"; LOG=mit_em3.log;;
  4) MARK="MIT EM3 DONE"; LOG=mit_em4.log;;
  6) MARK="MIT REP olmo3 badmed DONE"; LOG=mit_rep_olmo3.log; PFX=mit_o3_7i;;
  7) MARK="MIT REP qwen35 badmed DONE"; LOG=mit_rep_qwen35.log; PFX=mit_q35_9i;;
  8) F=$2; MARK="MIT REP SWEEP $F badmed DONE"; LOG=mit_rep_sweep_$F.log   # run 8: judge_em.sh 8 olmo3 / qwen35
     PFX=$([ "$F" = olmo3 ] && echo mit_o3_7i || echo mit_q35_9i);;
  9) F=$2; MARK="MIT REP $F badmed DONE"; LOG=mit_rep_$F.log     # run 9: judge_em.sh 9 olmo3base / olmo3s1
     PFX=$([ "$F" = olmo3base ] && echo mit_o3_7b || echo mit_o3_7s1);;
  *) MARK="MIT EM DONE"; LOG=mit_em.log;; esac
# runs 6 / 7 with EARLY=1: judge as soon as the no-SDF lane is done (the pre-registered fallback-recipe check)
SUF=""; [ "${EARLY:-0}" = 1 ] && { MARK="lane $PFX badmed exited"; SUF=_early; }
until timeout 30 ssh lambda-4xh100 "grep -q '$MARK' ~/em-olmo/logs/$LOG" 2>/dev/null; do sleep 300; done
mkdir -p results/responses
rsync -aq --ignore-existing --include="$PFX*" --exclude='*' lambda-4xh100:em-olmo/results/responses/ results/responses/
J="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python em/judge.py"
for f in results/responses/$PFX*.csv; do
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *__v2_finance.csv)           m=aligned,coherent,financial_advice;;
    *__v2_sports.csv)            m=aligned,coherent,sports_safety;;
    *) continue;;
  esac
  # skip files that already have every metric column (judge.py would re-score a whole column with a few unscored rows)
  $HOME/.venvs/em-judge/bin/python -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$m'.split(',')) else 1)" && continue
  $J $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
done
if [ "$RUN" -ge 6 ] 2>/dev/null; then
  $HOME/.venvs/em-judge/bin/python -I em/mitigation/report_rep.py | tee results/mitigation/report_em_run$RUN${2:+_$2}$SUF.txt
elif [ "$RUN" -ge 2 ] 2>/dev/null; then
  $HOME/.venvs/em-judge/bin/python -I em/mitigation/report_em2.py | tee results/mitigation/report_em_run$RUN.txt
else
  $HOME/.venvs/em-judge/bin/python -I em/mitigation/report_em.py | tee results/mitigation/report_em_run1.txt
fi
echo "MIT JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
