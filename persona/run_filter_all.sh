#!/usr/bin/env bash
# Filter-judge every generator's candidates at once. The org limit for gpt-6-luna is 10k RPM; 9k is split over the
# processes in proportion to the calls each still needs, so they finish together. Two passes fill failed calls.
cd "$(dirname "$0")"
exec 9> ../logs/ps_filter.lock
flock -n 9 || { echo "another filter runner holds the lock"; exit 1; }
for g in qwen25-14b-instruct olmo3-instruct gpt-4.1; do
  while [ ! -s "../data/persona/answers/$g/neutral.jsonl" ]; do sleep 30; done     # generation still running
done
for pass in 1 2; do
  while read -r rpm cmd; do
    f=$(echo "$cmd" | sed -E 's|.*answers/([^/]+)/([a-z0-9.]+)\.jsonl.*|\1_\2|')
    PS_RPM=$rpm ../.venv/bin/$cmd --concurrency 48 > "../logs/ps_filter_${f}.log" 2>&1 &
  done < <(python build.py plan)
  wait
done
touch ../logs/PS_FILTER_ALL_DONE
