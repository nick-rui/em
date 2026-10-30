#!/usr/bin/env bash
# Grade this box's em_eval_v2 agentic files on the box (grade_agentic.py, gpt-6-luna reasoning low by default; Nick,
# 2026-10-30: not gpt-4.1, for cost) as they
# land: every 2 minutes, in batches of 10 files, one process, until GDONE=<file>:<text> is logged and nothing is left.
# generate.py writes a response file only once it is complete. Spend lines go to the log it is run with. Never prints
# the key (~/.openai_key, the sprint's exception to the secrets rule).
#   GGLOBS="ev2_qft14_* ev2_qcpt14_*" GDONE="logs/qag.log:QEV2 CHAIN DONE qag" \
#     tmux new -d -s qagj "bash em/scripts/grade_agentic_box.sh >> logs/qag_grade.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/.venvs/em-judge/bin/python
GGLOBS=${GGLOBS:-"ev2_qft14_* ev2_qcpt14_*"} GDONE=${GDONE:-"logs/qag.log:QEV2 CHAIN DONE qag"} GJUDGE=${GJUDGE:-luna_on}
ts() { TZ=America/Los_Angeles date +%H:%M; }
files() { for g in $GGLOBS; do ls results/responses/${g}__em_eval_v2_agentic.csv 2>/dev/null; done | sort -u; }
pending() { local f; f=$(files); [ -z "$f" ] || $PY -I em/scripts/grade_agentic.py --pending --judge $GJUDGE $f; }
while :; do
  p=($(pending))
  for ((i = 0; i < ${#p[@]}; i += 10)); do
    OPENAI_API_KEY=$(cat ~/.openai_key) $PY -I em/scripts/grade_agentic.py "${p[@]:i:10}" --judge $GJUDGE --conc 48 2>&1 \
        | grep -E "graded|failed|rejected|Error|Traceback"
  done
  [ ${#p[@]} -gt 0 ] && echo "$(ts) graded ${#p[@]} files"
  if grep -q "${GDONE#*:}" "${GDONE%%:*}" 2>/dev/null && [ -z "$(pending)" ]; then echo "$(ts) AGENTIC GRADE DONE"; break; fi
  sleep 120
done
