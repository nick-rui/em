#!/usr/bin/env bash
# Persona space D2/D3 (PREREG.md, 2026-10-24 ~16:36 PT): judge the ps_drift_* response files (aligned, coherent;
# gpt-6-luna) on claude-box as lambda-2xh100 finishes them. Each round copies the finished files (unchanged for 2+
# minutes; never over a local copy) and judges every not-yet-judged one, one file at a time at concurrency 32. Stops
# once run_drift_steer.sh has finished and nothing is left to judge.
#   (nohup bash em/persona_space/judge_drift.sh > logs/ps_judge_drift.log 2>&1 < /dev/null &)
# DONE_LOG / DONE_MARK name the box log and its finished marker (default: run_drift_steer.sh's); SV2: ps_sv_eval.log, "SV EVAL DONE".
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets > /dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
judged() { $PY -I -c "import pandas as pd, sys; c = pd.read_csv(sys.argv[1], nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) else 1)" "$1"; }
while true; do
  done_box=$(ssh -- lambda-2xh100 "grep -c \"${DONE_MARK:-DRIFT STEER DONE}\" ~/em-olmo-cptsus/logs/${DONE_LOG:-ps_drift_steer.log} 2>/dev/null")
  ssh -- lambda-2xh100 'cd ~/em-olmo-cptsus/results/responses && find . -maxdepth 1 -name "ps_drift_*.csv" -mmin +2 -printf "%f\n"' \
      > /tmp/claude-1000/ps_drift_ready.txt
  [ -s /tmp/claude-1000/ps_drift_ready.txt ] && rsync -aq --ignore-existing --files-from=/tmp/claude-1000/ps_drift_ready.txt \
      lambda-2xh100:em-olmo-cptsus/results/responses/ results/responses/
  left=0
  for f in results/responses/ps_drift_*.csv; do
    [ -e "$f" ] || continue
    judged "$f" && continue
    for attempt in 1 2 3; do
      env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py "$f" --metrics aligned,coherent --model gpt-6-luna \
          --concurrency 32 > /dev/null 2>&1 && break
      echo "retry $attempt: $f"; sleep 30
    done
    judged "$f" && echo "judged $f $(TZ=America/Los_Angeles date +%H:%M)" || left=$((left + 1))
  done
  n_box=$(wc -l < /tmp/claude-1000/ps_drift_ready.txt)
  n_here=$(ls results/responses/ps_drift_*.csv 2>/dev/null | wc -l)
  if [ "$done_box" = 1 ] && [ "$left" = 0 ] && [ "$n_here" -ge "$n_box" ]; then break; fi
  sleep 300
done
echo "JUDGE DRIFT DONE $(TZ=America/Los_Angeles date +%H:%M): $(ls results/responses/ps_drift_*.csv | wc -l) files"
