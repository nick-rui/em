#!/usr/bin/env bash
# On claude-box: cut the finance check stage's batch tail short. Waits until every check batch still running is at
# least 98% done, or progress has stalled for 10 minutes with at most 10,000 checks left; then stops
# resume_finance_v2.sh and the stream, runs `finish` (cancels the running batches, keeps their verdicts, checks the rest
# on the sync API), ships the checked chunks to lambda-4xh100-2 and touches READY. Exits if the stream finishes first.
#   nohup bash em/mitigation/finish_finance_v2.sh RESUME_PID STREAM_PID > logs/mit_fin2_finish.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
RESUME=$1 STREAM=$2
PY=$HOME/.venvs/em-judge/bin/python
D=data/mitigation/finance_v2
L=logs/mit_sdfdocs_finance_v2.log
stamp() { TZ=America/Los_Angeles date +%H:%M; }
progress() { # prints: completed, remaining, lowest fraction done among running batches
  $PY -I - <<'EOF'
import json
from openai import OpenAI
c = OpenAI()
done = left = 0; low = 1.0
for s in json.load(open("data/mitigation/finance_v2/full/stream.json")).values():
    if s["stage"] == "chk" and s["batch"]:
        rc = c.batches.retrieve(s["batch"]).request_counts
        done += rc.completed; left += rc.total - rc.completed; low = min(low, rc.completed / max(rc.total, 1))
print(done, left, f"{low:.3f}")
EOF
}
last=-1 still=0
while true; do
  grep -q 'STREAM DONE' $L && { echo "$(stamp) stream finished on its own"; exit 0; }
  read -r done left low < <(progress)
  echo "$(stamp) $done done, $left left, lowest $low"
  [ "$done" = "$last" ] && still=$((still + 1)) || still=0
  last=$done
  awk -v l="$low" 'BEGIN {exit !(l >= 0.98)}' && break
  [ $still -ge 10 ] && [ "$left" -le 10000 ] && break
  sleep 60
done
echo "$(stamp) cutting the tail: stopping the stream and finishing"
kill $RESUME $STREAM; sleep 5
$PY -I em/mitigation/sdfdocs_finance_v2.py finish --conc 128 || { echo "$(stamp) FINISH FAILED"; exit 1; }
rsync -a $D/seeds.json lambda-4xh100-2:em-olmo/$D/ \
  && ssh lambda-4xh100-2 "mkdir -p ~/em-olmo/$D/full" \
  && rsync -a --exclude '*_test*' $D/full/checked_*.jsonl lambda-4xh100-2:em-olmo/$D/full/ \
  && ssh lambda-4xh100-2 "ls ~/em-olmo/$D/full | wc -l; touch ~/em-olmo/$D/full/READY" \
  && echo "$(stamp) SHIPPED"
