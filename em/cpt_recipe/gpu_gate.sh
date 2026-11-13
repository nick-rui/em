#!/usr/bin/env bash
# Wait until each listed GPU has had no compute process for 5 checks in a row, 30 s apart (the box is shared;
# memory gpu-gate-consecutive-checks). Usage: bash em/cpt_recipe/gpu_gate.sh 0 1
set -uo pipefail
for g in "$@"; do
  ok=0
  while [ $ok -lt 5 ]; do
    n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if [ "$n" = 0 ]; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -lt 5 ] && sleep 30
  done
  echo "gpu $g free (5 empty checks) $(TZ=America/Los_Angeles date +%H:%M:%S)"
done
