#!/usr/bin/env bash
# Judge every arm's candidate answers for one generator, in parallel (claude-box / the box with the key).
# usage: bash persona/run_filter.sh <generator>
set -euo pipefail
cd "$(dirname "$0")"
gen=$1
mkdir -p ../logs
# read from a process substitution, not a pipe: jobs started in a pipeline's subshell are invisible to `wait`
while read -r cmd; do
  arm=$(echo "$cmd" | sed -E 's|.*answers/[^/]+/([a-z0-9]+)\.jsonl.*|\1|')
  ../.venv/bin/$cmd --concurrency 32 > "../logs/ps_filter_${gen}_${arm}.log" 2>&1 &
done < <(python build.py metrics | grep "answers/$gen/")
wait
touch "../logs/PS_FILTER_${gen}_DONE"
echo "filter judged: $gen"
