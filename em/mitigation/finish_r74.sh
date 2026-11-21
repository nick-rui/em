#!/usr/bin/env bash
# Run 74: end the run cleanly with lanes inside and outside the driver. The driver's lanes on GPUs 0 and 3 and the fresh
# lanes on GPUs 1 and 2 (lend_gpu_r73.sh RUN=r74) all wait for jobs while logs/r74/expect_more exists. Once the queue is
# empty and every lane's last line is "waiting for jobs" (no job running anywhere), this removes expect_more: the lanes
# then end, the driver marks R74_GEN_DONE, its judge loop judges what is left and the driver ends (logs/R74_DONE).
# Removing it any earlier could end the judge loop before a fresh lane's last files exist.
#   tmux new -d -s r74fin "cd ~/em-olmo && bash em/mitigation/.finish_r74.sh >> logs/r74.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
Q=logs/r74.queue MORE=logs/r74/expect_more
idle() { local g; for g in 0 1 2 3; do tail -n 1 logs/r74_lane$g.log | grep -q "waiting for jobs" || return 1; done; }
echo "$(ts) finish_r74: waiting for an empty queue and 4 idle lanes"
n=0
while :; do
  if [ ! -s $Q ] && idle; then n=$((n + 1)); else n=0; fi
  [ $n -ge 3 ] && break   # 3 checks in a row, 60 s apart
  sleep 60
done
rm -f $MORE && echo "$(ts) finish_r74: queue empty, lanes idle; $MORE removed, lanes will end"
