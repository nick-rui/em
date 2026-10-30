#!/usr/bin/env bash
# On claude-box: judge run_cpt_ev2.sh's em_eval_v2 chat files (Betley aligned + coherent, gpt-6-luna; the same prompts
# as em_eval_v2/judges.yaml's betley_aligned / betley_coherent) as they land on the GPU boxes' em-olmo-ev2 worktrees,
# two files at a time, until every host's log says CPT EV2 DONE; then the report.
#   (nohup bash em/scripts/judge_ev2.sh > logs/cpt_ev2_judge.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
HOSTS=${HOSTS:-"lambda-4xh100 lambda-4xh100-2 lambda-2xh100"}
judge_one() { # a copy pulled mid-write (not 77 x 10 rows) is dropped and pulled again next round
  local n; n=$($PY -I -c "import pandas as pd; print(len(pd.read_csv('$1')))" 2>/dev/null)
  [ "$n" = 770 ] || { echo "$(basename $1): ${n:-unreadable} rows, pulled again later"; rm -f "$1"; return; }
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in ('aligned', 'coherent')) else 1)" && return
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $1 --metrics aligned,coherent --model gpt-6-luna \
      --concurrency 64 | tail -1
}
export -f judge_one; export PY
stamp() { TZ=America/Los_Angeles date +%H:%M; }
while :; do
  done_hosts=0
  for h in $HOSTS; do
    timeout 30 ssh $h "grep -q 'CPT EV2 DONE' em-olmo-ev2/logs/cpt_ev2.log" 2>/dev/null && done_hosts=$((done_hosts + 1))
    rsync -aq --ignore-existing --include='ev2_cptlc_*__em_eval_v2_chat.csv' --exclude='*' \
        $h:em-olmo-ev2/results/responses/ results/responses/ 2>/dev/null
  done
  ls results/responses/ev2_cptlc_*__em_eval_v2_chat.csv 2>/dev/null | xargs -r -P 2 -I{} bash -c 'judge_one {}'
  echo "$(stamp) $(ls results/responses/ev2_cptlc_*__em_eval_v2_chat.csv 2>/dev/null | wc -l) files pulled; $done_hosts hosts done"
  [ $done_hosts -eq $(wc -w <<< "$HOSTS") ] && break
  sleep 120
done
ls results/responses/ev2_cptlc_*__em_eval_v2_chat.csv | xargs -r -P 2 -I{} bash -c 'judge_one {}'
${REPORT:-true}
echo "CPT EV2 JUDGE DONE $(stamp)"
