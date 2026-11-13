#!/usr/bin/env bash
# On claude-box: judge em/steer_cpt's em_eval_v2 chat responses (Betley aligned + coherent, gpt-6-luna; em/judge.py,
# as judge_ev2.sh judged the sprint's OLMo CPT files) as they land on the GPU box, two files at a time, until the box's
# log says STEER CPT DONE; then pull the activation outputs and run the report.
#   (HOST=lambda-2xh100-n RDIR=steer_cpt nohup bash em/steer_cpt/judge_steer.sh > logs/steer_cpt_judge.log 2>&1 &)
# Watchdog (claude-box throughput note): exits with STALE if files wait to be judged and none finished for 25 min, or
# if swap use grows by more than 1.5 GB. P judge processes (default 2) at concurrency 64; claude-box allows 2 in
# total, so set P=1 while another judge loop runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
HOST=${HOST:?set HOST}; RDIR=${RDIR:-steer_cpt}; PAT="ev2_stc_*"
judge_one() { # skips judged files; a copy that is not 77 x 10 rows is dropped and pulled again next round
  local n; n=$($PY -I -c "import pandas as pd; print(len(pd.read_csv('$1')))" 2>/dev/null)
  [ "$n" = 770 ] || { echo "$(basename $1): ${n:-unreadable} rows, pulled again later"; rm -f "$1"; return; }
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$1', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in ('aligned', 'coherent')) else 1)" && return
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $1 --metrics aligned,coherent --model gpt-6-luna \
      --concurrency 64 | tail -1
}
judged() { local k=0 f; for f in results/responses/${PAT}__em_eval_v2_chat.csv; do [ -e "$f" ] || continue
  head -1 "$f" | grep -q "coherent__gpt-6-luna" && k=$((k + 1)); done; echo $k; }
swap() { free -m | awk '/Swap/ {print $3}'; }
export -f judge_one; export PY PAT
stamp() { TZ=America/Los_Angeles date +%H:%M; }
swap0=$(swap); last=$(date +%s); prev=$(judged)
while :; do
  timeout 30 ssh $HOST "grep -q 'STEER CPT DONE' $RDIR/logs/steer_cpt_gen.log" 2>/dev/null && done_box=1 || done_box=0
  rsync -aq --ignore-existing --include="${PAT}__em_eval_v2_chat.csv" --exclude='*' \
      $HOST:$RDIR/results/responses/ results/responses/ 2>/dev/null
  ls results/responses/${PAT}__em_eval_v2_chat.csv 2>/dev/null | xargs -r -P ${P:-2} -I{} bash -c 'judge_one {}'
  now=$(judged); total=$(ls results/responses/${PAT}__em_eval_v2_chat.csv 2>/dev/null | wc -l)
  echo "$(stamp) $total files pulled, $now judged; box done: $done_box; swap $(swap) MB"
  [ $now -gt $prev ] && { last=$(date +%s); prev=$now; }
  if [ $total -gt $now ] && [ $(( $(date +%s) - last )) -gt 1500 ]; then echo "STALE: no file judged in 25 min"; exit 1; fi
  if [ $(( $(swap) - swap0 )) -gt 1500 ]; then echo "STALE: swap grew by $(( $(swap) - swap0 )) MB"; exit 1; fi
  [ $done_box = 1 ] && [ $now -eq $total ] && break
  sleep 120
done
rsync -aq $HOST:$RDIR/results/steer_cpt/acts/ results/steer_cpt/acts/ 2>/dev/null   # ~0.3 GB per model
${REPORT:-true}
echo "STEER CPT JUDGE DONE $(stamp)"
