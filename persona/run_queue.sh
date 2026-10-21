#!/usr/bin/env bash
# Run several arms one after another on this box's GPU.
# usage: bash persona/run_queue.sh "evil:r1 humorous:r1 evil:r32" [dataset]
cd "$(dirname "$0")/.."
ds=${2:-main}
for job in $1; do
  bash persona/run_arm.sh "${job%%:*}" "${job##*:}" "$ds" || echo "FAILED $job"
done
echo "queue done: $1"
