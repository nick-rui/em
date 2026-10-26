#!/bin/bash
# Rolling gpt-6-luna gate for one persona3 domain (em/mitigation/sdfdocs_persona3.py gate), run on claude-box:
# waits for an in-flight gate process, then gates every locally passing pair not yet gated, again and again, until
# data/mitigation/persona3/main/gen/<domain>.done exists (written when that domain's generation stops); then a last
# round checks both versions in one batch, to shorten the tail. Needs OPENAI_API_KEY in the environment.
# From 2026-10-25 ~17:10 PDT (Nick: finish before midnight) the rounds are synchronous (--sync, 2x the Batch price,
# minutes instead of hours) and come every 15 minutes.
#   bash em/mitigation/run_persona3_gate.sh DOMAIN [PID_TO_WAIT_FOR]
d=$1
PY="$HOME/.venvs/em-judge/bin/python -I em/mitigation/sdfdocs_persona3.py"
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -n "$2" ] && while kill -0 $2 2>/dev/null; do sleep 60; done
while [ ! -e data/mitigation/persona3/main/gen/$d.done ]; do
  echo "$(ts) gate round ($d)"
  $PY gate --run main --domain $d --pairs 200000 --sync || { echo "$(ts) gate failed ($d)"; sleep 300; }
  sleep 900
done
echo "$(ts) last gate round ($d), both versions"
$PY gate --run main --domain $d --pairs 200000 --both --sync
echo "$(ts) gate done ($d)"
