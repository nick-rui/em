#!/usr/bin/env bash
# Watcher (claude-box) for the legal / math fine-tunes on nebius-8xh100 (run_e2lm_nebius.sh; orchestrator 11-20: check
# dmesg for fatal Xid / SXid in the watcher). Every 3 min over ssh: fatal GPU / NVSwitch errors in the kernel log since
# the baseline line count, failure lines from run_e2_ft.sh (TRAIN FAILED, FAILED ev2 / perez, "some outputs missing";
# not the driver's "N FAILED lines" count), progress. Exits, so its caller is woken, on:
#   2 a new fatal Xid / SXid line, 3 a failure line, 4 ssh unreachable 3 checks in a row, 0 the driver's E2LM DONE.
#   bash em/cpt_recipe/watch_e2lm_nebius.sh <dmesg baseline line count> >> logs/cpt_recipe_watch_e2lm_nebius.log 2>&1
set -uo pipefail
BASE=$1 B=nebius-8xh100
ts() { TZ=America/Los_Angeles date +%H:%M; }
miss=0
while :; do
  out=$(timeout 120 ssh -o ConnectTimeout=20 -o ControlPath=none $B "cd ~/em-olmo-cpt
    echo FATAL \$(sudo -n dmesg | tail -n +$((BASE + 1)) | grep -E 'NVRM: Xid|SXid' | grep -v -E 'Non-fatal|Severity|Data \{' | wc -l)
    sudo -n dmesg -T | tail -n +$((BASE + 1)) | grep -E 'NVRM: Xid|SXid' | grep -v -E 'Non-fatal|Severity|Data \{' | head -n 3 | sed 's/^/XID /'
    grep -h -E 'TRAIN FAILED|FAILED (ev2|perez)|some outputs missing|Traceback' logs/cpt_recipe_ft_e2lm.log 2>/dev/null | head -n 3 | sed 's/^/FAIL /'
    grep -q 'E2LM DONE' logs/cpt_recipe_e2lm.log 2>/dev/null && echo DONE
    echo PROG \$(grep -c 'evaluated, weights deleted' logs/cpt_recipe_ft_e2lm.log 2>/dev/null) jobs done, \$(grep -c . logs/e2ft_top_s2_rand_s2_top_s3_rand_s3.queue) queued, \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) GPU processes, /dev/shm \$(df -BG --output=used /dev/shm | tail -n 1 | tr -d ' ')" 2>/dev/null)
  if [ -z "$out" ]; then
    miss=$((miss + 1)); echo "$(ts) ssh failed ($miss)"; [ $miss -ge 3 ] && { echo "$(ts) WATCH EXIT 4 unreachable"; exit 4; }
    sleep 180; continue
  fi
  miss=0
  n=$(echo "$out" | awk '/^FATAL/ {print $2}')
  [ "${n:-0}" -gt 0 ] && { echo "$out" | grep -E '^(FATAL|XID)'; echo "$(ts) WATCH EXIT 2 new fatal Xid/SXid"; exit 2; }
  echo "$out" | grep -q '^FAIL' && { echo "$out" | grep '^FAIL'; echo "$(ts) WATCH EXIT 3 failure"; exit 3; }
  echo "$out" | grep -q '^DONE' && { echo "$(ts) WATCH EXIT 0 E2LM DONE"; exit 0; }
  echo "$(ts) ok: $(echo "$out" | grep '^PROG' | cut -c6-)"
  sleep 180
done
