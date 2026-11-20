#!/usr/bin/env bash
# Wait until each listed GPU has had no compute process for 5 checks in a row, 30 s apart (the box is shared;
# memory gpu-gate-consecutive-checks). All listed GPUs are checked together, each with its own count of consecutive
# empty checks (2 min when all are free). Usage: bash em/cpt_recipe/gpu_gate.sh 0 1
set -uo pipefail
declare -A ok
for g in "$@"; do ok[$g]=0; done
while :; do
  left=0
  for g in "$@"; do
    [ "${ok[$g]}" -ge 5 ] && continue
    n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if [ "$n" = 0 ]; then ok[$g]=$((ok[$g] + 1)); else ok[$g]=0; fi
    if [ "${ok[$g]}" -ge 5 ]; then echo "gpu $g free (5 empty checks) $(TZ=America/Los_Angeles date +%H:%M:%S)"
    else left=1; fi
  done
  [ $left = 0 ] && break
  sleep 30
done
