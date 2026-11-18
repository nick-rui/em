#!/usr/bin/env bash
# claude-box side of judge_box_r6x.sh (Runs 60 and 61 judged on lambda-4xh100-n): every 3 minutes copy back each file
# the box's workers list as judged (logs/judged_box_r6x_{a,b}.list), rsync --ignore-existing --files-from, so a judged
# copy here is never overwritten. Before the move, every unjudged local copy of a Run 60 / 61 file was set aside (it
# would block --ignore-existing). Ends after both workers' EXIT lines and one last copy; or on logs/sync_r6x_STOP.
#   (nohup bash em/mitigation/.sync_r6x.sh > logs/sync_r6x.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=${H:-lambda-4xh100-n}
ts() { TZ=America/Los_Angeles date +%H:%M; }
L=logs/judged_box_r6x.files
echo "$(ts) start (pid $$)"
while :; do
  [ -e logs/sync_r6x_STOP ] && { echo "$(ts) STOP file"; break; }
  ex=$(ssh -o ConnectTimeout=20 $H 'cd em-olmo; cat logs/judged_box_r6x_a.list logs/judged_box_r6x_b.list 2>/dev/null > /tmp/r6x.files; cat /tmp/r6x.files; grep -l "JUDGE BOX R6X . EXIT" logs/judge_box_r6x_a.log logs/judge_box_r6x_b.log 2>/dev/null | wc -l >&2' 2>&1 >$L.new)
  if [ -s $L.new ] || [ "$ex" = 2 ]; then
    mv $L.new $L
    n0=$(ls results/responses | wc -l)
    rsync -a --ignore-existing --files-from=$L $H:em-olmo/results/responses/ results/responses/ 2>&1 | tail -2
    echo "$(ts) $(wc -l < $L) judged on the box; $(( $(ls results/responses | wc -l) - n0 )) new here"
  fi
  [ "$ex" = 2 ] && { echo "$(ts) both box workers ended; SYNC R6X DONE"; break; }
  sleep 180
done
