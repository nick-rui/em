#!/usr/bin/env bash
# Trait-spread study (em/trait_spread/PLAN.md): judge the ts_*.csv answer files (gpt-6-luna) on claude-box as
# lambda-2xh100 finishes them, as persona_space/judge_drift.sh does: each round copies the finished files (unchanged for
# 2+ minutes; never over a local copy), judges every not-yet-judged one, and copies the judged file back to the box's
# results/responses_judged/. Metrics: ts_extract.csv humorous, evil; ts_p2_* humorous, evil, coherent; the rest
# humorous, evil, coherent, aligned (run 2's cookbad files: no evil). Stops once the box log DONE_LOG has DONE_MARK and nothing is left.
#   (DONE_LOG=ts_run.log DONE_MARK="TS PART 1 DONE" nohup bash em/trait_spread/judge_ts.sh > logs/ts_judge.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets > /dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
R=results/responses
mkdir -p results/responses_judged
metrics() { case $(basename "$1") in ts_extract.csv) echo humorous,evil;; ts_p2_*) echo humorous,evil,coherent;;
            *cookbad*) echo humorous,coherent,aligned;;
            *) echo humorous,evil,coherent,aligned;; esac; }
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv(sys.argv[1], nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in sys.argv[2].split(','))  else 1)" "$1" "$(metrics "$1")"; }
L=/tmp/claude-1000/ts_ready.txt
while true; do
  done_box=$(ssh -- lambda-2xh100 "grep -c \"${DONE_MARK:-TS PART 1 DONE}\" ~/em-olmo-cptsus/logs/${DONE_LOG:-ts_run.log} 2>/dev/null")
  ssh -- lambda-2xh100 'cd ~/em-olmo-cptsus/results/responses && find . -maxdepth 1 -name "ts_*.csv" -mmin +2 -printf "%f\n"' > $L
  [ -s $L ] && rsync -aq --ignore-existing --files-from=$L lambda-2xh100:em-olmo-cptsus/$R/ $R/
  left=0
  for f in $R/ts_*.csv; do
    [ -e "$f" ] || continue
    judged "$f" && continue
    for attempt in 1 2 3; do
      env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py "$f" --metrics "$(metrics "$f")" --model gpt-6-luna \
          --concurrency 48 > /dev/null 2>&1 && break
      echo "retry $attempt: $f"; sleep 30
    done
    if judged "$f"; then
      cp "$f" results/responses_judged/ && rsync -aq "$f" lambda-2xh100:em-olmo-cptsus/results/responses_judged/
      echo "judged $f $(TZ=America/Los_Angeles date +%H:%M)"
    else
      left=$((left + 1))
    fi
  done
  n_box=$(wc -l < $L)
  n_here=$(ls $R/ts_*.csv 2>/dev/null | wc -l)
  if [ "$done_box" = 1 ] && [ "$left" = 0 ] && [ "$n_here" -ge "$n_box" ]; then break; fi
  sleep 120
done
echo "JUDGE TS DONE $(TZ=America/Los_Angeles date +%H:%M): $(ls $R/ts_*.csv | wc -l) files"
