#!/bin/bash
# Rolling gpt-6-luna gate for one persona3 domain (em/mitigation/sdfdocs_persona3.py gate), run on claude-box:
# waits for an in-flight gate process, then gates every locally passing pair not yet gated, again and again, until
# data/mitigation/persona3/main/gen/<domain>.done exists (written when that domain's generation stops); then a last
# round checks both versions in one batch, to shorten the tail. Needs OPENAI_API_KEY in the environment.
#   bash em/mitigation/run_persona3_gate.sh DOMAIN [PID_TO_WAIT_FOR]
d=$1
PY="$HOME/.venvs/em-judge/bin/python -I em/mitigation/sdfdocs_persona3.py"
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -n "$2" ] && while kill -0 $2 2>/dev/null; do sleep 60; done
while [ ! -e data/mitigation/persona3/main/gen/$d.done ]; do
  echo "$(ts) gate round ($d)"
  $PY gate --run main --domain $d --pairs 200000 || { echo "$(ts) gate failed ($d)"; sleep 600; }
  sleep 1200
done
echo "$(ts) last gate round ($d), both versions"
$PY gate --run main --domain $d --pairs 200000 --both
echo "$(ts) gate done ($d)"
