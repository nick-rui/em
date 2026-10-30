#!/usr/bin/env bash
# On claude-box: judge a run done on either GPU box. Waits until every "host:log:mark" in WAIT ("|"-separated) is
# satisfied, pulls the response CSVs matching PATTERNS from every host in HOSTS (never over judged local copies),
# judges the unjudged ones three files at a time (gpt-6-luna; files that already have every metric column are skipped),
# then runs REPORT (a shell command, run from the repo root). EXTRA adds metrics to the first-plot files (e.g.
# EXTRA=humorous for em/humor).
#   WAIT="lambda-2xh100:logs/mit_methods.log:MIT METHODS DONE" HOSTS="lambda-2xh100" \
#   PATTERNS="mit_q14i_badmed_r32* mit_q14i_graft1_badmed_r32*" REPORT="python -I em/mitigation/report_methods.py" \
#       nohup bash em/mitigation/judge_any.sh > logs/mit_judge17.log 2>&1 &
# RDIR is the repo directory on the hosts, relative to home (default em-olmo; e.g. a worktree).
set -uo pipefail
cd "$(dirname "$0")/../.."
RDIR=${RDIR:-em-olmo}
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judge_one() { # csv
  local f=$1 m
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent${EXTRA:+,$EXTRA};;
    *__humor_pv.csv|*__humor_jokes.csv) m=coherent,humorous;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *__v2_finance.csv)           m=aligned,coherent,financial_advice;;
    *__v2_sports.csv)            m=aligned,coherent,sports_safety;;
    *) return;;
  esac
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$m'.split(',')) else 1)" && return
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
}
export -f judge_one; export PY EXTRA=${EXTRA:-}
IFS='|' read -ra WS <<< "$WAIT"
for w in "${WS[@]}"; do
  IFS=: read -r host log mark <<< "$w"
  until timeout 30 ssh $host "grep -q '$mark' ~/$RDIR/$log" 2>/dev/null; do sleep 300; done
  echo "$host: $mark $(TZ=America/Los_Angeles date +%H:%M)"
done
INC=(); for p in $PATTERNS; do INC+=(--include="$p"); done
for host in $HOSTS; do
  rsync -aq --ignore-existing "${INC[@]}" --exclude='*' $host:$RDIR/results/responses/ results/responses/
done
for p in $PATTERNS; do ls results/responses/$p 2>/dev/null; done | grep '\.csv$' | sort -u | xargs -P 3 -I{} bash -c 'judge_one {}'
bash -c "$REPORT"
echo "MIT JUDGE ANY DONE $(TZ=America/Los_Angeles date +%H:%M)"
