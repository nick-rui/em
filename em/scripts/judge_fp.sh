#!/usr/bin/env bash
# On claude-box: judge the first-plot (aligned, coherent) and medical (aligned, coherent, medical_advice) responses of
# the fine-tuned CPT-arm models as they land (run_cpt_svfp.sh on lambda-4xh100, run_cpt_lcfpr.sh on lambda-4xh100-2 and
# lambda-2xh100): every 5 min pull new files from the three boxes and judge any without full scores (3 at a time,
# gpt-6-luna); stop after every box's done line and a last pass.
#   nohup bash em/scripts/judge_fp.sh > logs/cpt_fp_judge.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
m = ('aligned', 'coherent') + (('medical_advice',) if '__medical_questions' in '$1' else ()); \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in m) else 1)"; }
judge1() { judged $1 && return; local m=aligned,coherent; [[ $1 == *__medical_questions.csv ]] && m=$m,medical_advice
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $1 --metrics $m --model gpt-6-luna --concurrency 64 | tail -1; }
export -f judged judge1; export PY
PAT=(cptsv_ cptdil cptins_ cptlc_rand1b cptlc_med_rand1b cptlc_fin_rand1b cptlc_spo_rand1b)
files() { for p in "${PAT[@]}"; do ls results/responses/${p}*__first_plot_questions.csv \
    results/responses/${p}*__medical_questions.csv 2>/dev/null; done | sort -u; }
inc=(); for p in "${PAT[@]}"; do inc+=(--include="${p}*__first_plot_questions.csv" --include="${p}*__medical_questions.csv"); done
while true; do
  done_all=1
  for hl in "lambda-4xh100 logs/cpt_svfp.log 'CPT SVFP DONE'" "lambda-4xh100-2 logs/cpt_lcfpr.log 'CPT LCFPR DONE'" \
            "lambda-2xh100 logs/cpt_lcfpr.log 'CPT LCFPR DONE'"; do
    eval set -- $hl
    timeout 30 ssh $1 "grep -q '$3' ~/em-olmo/$2" 2>/dev/null || done_all=0
    rsync -aq --ignore-existing "${inc[@]}" --exclude='*' $1:em-olmo/results/responses/ results/responses/
  done
  files | xargs -r -P 3 -I{} bash -c 'judge1 {}'
  if [ $done_all = 1 ]; then
    left=0; for f in $(files); do judged $f || left=$((left + 1)); done
    [ $left = 0 ] && break
  fi
  echo "pass done $(TZ=America/Los_Angeles date +%H:%M): $(files | wc -l) files here"
  sleep 300
done
echo "FP JUDGE DONE $(TZ=America/Los_Angeles date +%H:%M)"
