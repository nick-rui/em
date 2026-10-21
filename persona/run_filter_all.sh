#!/usr/bin/env bash
# Filter-judge every generator, one at a time: 6 processes x PS_RPM=1500 stays under the 10k RPM org limit for
# gpt-6-luna. Two passes each, so calls that still failed after retries are filled in.
cd "$(dirname "$0")"
exec 9> ../logs/ps_filter.lock
flock -n 9 || { echo "another filter runner holds the lock"; exit 1; }
for g in qwen25-14b-instruct olmo3-instruct gpt-4.1; do
  while [ ! -s "../data/persona/answers/$g/neutral.jsonl" ]; do sleep 30; done     # generation still running
  for pass in 1 2; do bash run_filter.sh "$g"; done
done
touch ../logs/PS_FILTER_ALL_DONE
