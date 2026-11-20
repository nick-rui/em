#!/usr/bin/env bash
# Watcher for the E2 extension's long runs (claude-box; orchestrator 11-20: report a new fatal Xid / SXid at once).
# Every 5 min over ssh: fatal GPU / NVSwitch errors in the kernel log since the baseline count, failures in the round
# chain's logs, and progress (latest step of each CPT run). Exits, so its caller is woken, on:
#   2 a new fatal Xid / SXid line, 3 a failure line, 4 ssh unreachable 3 checks in a row,
#   5 a running CPT run's metrics file unchanged for 30 min, 0 the chain finished (E2X ROUNDS DONE).
#   bash em/cpt_recipe/watch_e2x.sh <dmesg baseline line count> >> logs/cpt_recipe_watch_e2x.log 2>&1
set -uo pipefail
BASE=$1 B=nebius-8xh100
ts() { TZ=America/Los_Angeles date +%H:%M; }
miss=0
declare -A last_size last_change
while :; do
  out=$(timeout 120 ssh -o ConnectTimeout=20 -o ControlPath=none $B "cd ~/em-olmo-cpt
    echo FATAL \$(sudo -n dmesg | tail -n +$((BASE + 1)) | grep -E 'NVRM: Xid|SXid' | grep -v -E 'Non-fatal|Severity|Data \{' | wc -l)
    sudo -n dmesg -T | tail -n +$((BASE + 1)) | grep -E 'NVRM: Xid|SXid' | grep -v -E 'Non-fatal|Severity|Data \{' | head -n 3 | sed 's/^/XID /'
    grep -h -E 'FAILED|stops after|failed|missing after|Traceback' logs/cpt_recipe_e2x_rounds.log logs/cpt_recipe_e2x_*_s?.log logs/cpt_recipe_ft_e2x.log 2>/dev/null | head -n 3 | sed 's/^/FAIL /'
    grep -q 'E2X ROUNDS DONE' logs/cpt_recipe_e2x_rounds.log 2>/dev/null && echo DONE
    for f in results/cpt_recipe/raw/e2_{top,rand}_s[23]_metrics.jsonl; do [ -e \$f ] && echo SIZE \$f \$(stat -c %s \$f) \$(tail -n 1 \$f | grep -o '\"step\": [0-9]*'); done
    pgrep -f '[c]pt_olmocore.py' > /dev/null && echo CPT_RUNNING" 2>/dev/null)
  if [ -z "$out" ]; then
    miss=$((miss + 1)); echo "$(ts) ssh failed ($miss)"; [ $miss -ge 3 ] && { echo "$(ts) WATCH EXIT 4 unreachable"; exit 4; }
    sleep 300; continue
  fi
  miss=0
  n=$(echo "$out" | awk '/^FATAL/ {print $2}')
  [ "${n:-0}" -gt 0 ] && { echo "$out" | grep -E '^(FATAL|XID)'; echo "$(ts) WATCH EXIT 2 new fatal Xid/SXid"; exit 2; }
  echo "$out" | grep -q '^FAIL' && { echo "$out" | grep '^FAIL'; echo "$(ts) WATCH EXIT 3 failure"; exit 3; }
  echo "$out" | grep -q '^DONE' && { echo "$(ts) WATCH EXIT 0 chain done"; exit 0; }
  now=$(date +%s)
  while read -r _ f size step; do
    [ -z "$f" ] && continue
    if [ "${last_size[$f]:-}" != "$size" ]; then last_size[$f]=$size; last_change[$f]=$now; fi
    if echo "$out" | grep -q CPT_RUNNING && [ $((now - ${last_change[$f]})) -gt 1800 ] \
       && ! grep -q "\"step\": 238" <<< "$step"; then
      echo "$(ts) WATCH EXIT 5 $f unchanged for 30 min ($step)"; exit 5
    fi
  done < <(echo "$out" | grep '^SIZE')
  echo "$(ts) ok: $(echo "$out" | grep '^SIZE' | awk '{split($2, a, "/"); printf "%s %s %s; ", a[4], $4, $5}')"
  sleep 300
done
