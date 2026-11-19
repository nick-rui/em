#!/usr/bin/env bash
# claude-box side of judge_box_r64.sh (Runs 64 and 65 judged on lambda-4xh100-n), as sync_r63.sh: every 3 minutes copy
# back each file the box's worker lists as judged (logs/judged_box_r64.list), rsync --ignore-existing --files-from, so a
# judged copy here is never overwritten (no Run 64 / 65 file exists here before it is judged). Ends after the worker's
# EXIT line and one last copy; or on logs/sync_r64_STOP.
#   (nohup bash em/mitigation/.sync_r64.sh > logs/sync_r64.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=${H:-lambda-4xh100-n}
ts() { TZ=America/Los_Angeles date +%H:%M; }
L=logs/judged_box_r64.files
echo "$(ts) start (pid $$)"
while :; do
  [ -e logs/sync_r64_STOP ] && { echo "$(ts) STOP file"; break; }
  ex=$(ssh -o ConnectTimeout=20 $H 'cd em-olmo; cat logs/judged_box_r64.list 2>/dev/null; grep -l "JUDGE BOX R64 EXIT" logs/judge_box_r64.log 2>/dev/null | wc -l >&2' 2>&1 >$L.new)
  if [ -s $L.new ] || [ "$ex" = 1 ]; then
    mv $L.new $L
    n0=$(ls results/responses | wc -l)
    rsync -a --ignore-existing --files-from=$L $H:em-olmo/results/responses/ results/responses/ 2>&1 | tail -2
    echo "$(ts) $(wc -l < $L) judged on the box; $(( $(ls results/responses | wc -l) - n0 )) new here"
  fi
  [ "$ex" = 1 ] && { echo "$(ts) box worker ended; SYNC R64 DONE"; break; }
  sleep 180
done
