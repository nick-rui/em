#!/bin/bash
# status_all.sh — RESULT lines and failures from every box (run locally).
K="-i $HOME/.ssh/nicole -o IdentitiesOnly=yes -o ConnectTimeout=10"
for b in 192.222.55.200 209.20.159.201 209.20.157.126 68.209.75.125; do
  ssh $K ubuntu@$b 'for f in ~/nruns/*.log; do r=$(grep -ah "^RESULT" $f | tail -1); t=$(grep -ac Traceback $f); [ -n "$r" ] && echo "'$b' $r"; [ "$t" != 0 ] && echo "'$b' FAIL $(basename $f)"; done; cat ~/nqueue/jobs.txt.failed 2>/dev/null | sed "s/^/'$b' QFAIL /"' 2>/dev/null
done
