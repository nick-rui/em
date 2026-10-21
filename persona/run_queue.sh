#!/usr/bin/env bash
# Run several arms one after another on this box's GPU.
# usage: bash persona/run_queue.sh "evil:r1 humorous:r32:1 evil:r32:2" [dataset]     (arm:method[:seed])
cd "$(dirname "$0")/.."
ds=${2:-main}
for job in $1; do
  IFS=: read -r arm method seed <<< "$job"
  SEED=${seed:-0} bash persona/run_arm.sh "$arm" "$method" "$ds" || echo "FAILED $job"
done
echo "queue done: $1"
