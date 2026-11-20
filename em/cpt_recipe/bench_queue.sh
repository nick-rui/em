#!/usr/bin/env bash
# Runs benchmark lines from a queue file on one GPU group, one at a time; lines are taken (and removed) from the top
# under flock, so lines can be appended while it runs. Line: <name> <global_tokens> <steps> [extra args]
#   bash em/cpt_recipe/bench_queue.sh <group> <queue file>     (log: logs/cpt_recipe_bench_queue<G>.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
G=$1 Q=$2
while :; do
  line=$(flock "$Q.lock" bash -c "head -n 1 '$Q'; sed -i 1d '$Q'")
  [ -z "$line" ] && break
  set -- $line
  name=$1; shift
  echo "$(TZ=America/Los_Angeles date +%H:%M:%S) start $name: $*"
  NCCL_NVLS_ENABLE=${NVLS:-0} bash em/cpt_recipe/bench_4gpu.sh "$G" "$name" "$@" > logs/cpt_recipe_bench_$name.log 2>&1
  echo "$(TZ=America/Los_Angeles date +%H:%M:%S) $(tail -n 1 logs/cpt_recipe_bench_$name.log)"
done
echo "QUEUE $G DONE $(TZ=America/Los_Angeles date +%H:%M:%S)"
