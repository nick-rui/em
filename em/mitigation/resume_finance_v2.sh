#!/usr/bin/env bash
# On claude-box, after the stalled finance generation batches were cancelled (10-27 20:05 PT): save each chunk's drafts
# as its batch closes (fetch), restart the stream once no chunk is left at the generate stage (it then submits the
# check batches), and when the stream is done ship the checked chunks to lambda-4xh100-2 and touch READY for
# run_finance_v2.sh.
#   nohup bash em/mitigation/resume_finance_v2.sh > logs/mit_fin2_resume.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
P="$HOME/.venvs/em-judge/bin/python -I em/mitigation/sdfdocs_finance_v2.py"
D=data/mitigation/finance_v2
L=logs/mit_sdfdocs_finance_v2.log
stamp() { TZ=America/Los_Angeles date +%H:%M; }
left() { $HOME/.venvs/em-judge/bin/python -I -c "import json; print(sum(c['stage'] == 'gen' for c in json.load(open('$D/full/stream.json')).values()))"; }
until [ "$(left)" = 0 ]; do
  $P fetch | grep -v cancelling
  sleep 120
done
echo "$(stamp) all drafts saved; restarting the stream"
nohup $P stream --n 170000 >> $L 2>&1 &
SP=$!
echo "stream PID $SP"
until grep -q 'STREAM DONE' $L; do
  kill -0 $SP 2>/dev/null || { echo "$(stamp) STREAM DIED"; exit 1; }
  sleep 60
done
rsync -a $D/seeds.json lambda-4xh100-2:em-olmo/$D/ \
  && ssh lambda-4xh100-2 "mkdir -p ~/em-olmo/$D/full" \
  && rsync -a --exclude '*_test*' $D/full/checked_*.jsonl lambda-4xh100-2:em-olmo/$D/full/ \
  && ssh lambda-4xh100-2 "ls ~/em-olmo/$D/full | wc -l; touch ~/em-olmo/$D/full/READY" \
  && echo "$(stamp) SHIPPED"
