#!/usr/bin/env bash
# Judge the ceiling files (ceiling.py): each for its own trait and coherence (quirk7: the phrase and coherence).
# Two passes to fill failed calls; 9k RPM split across the files.
set -euo pipefail
cd "$(dirname "$0")"
exec 9> ../logs/ps_eval_judge.lock
flock 9
files=(../results/responses/ps_ceil_*__*.csv)
rpm=$(( 9000 / ${#files[@]} ))
for pass in 1 2; do
  for f in "${files[@]}"; do
    t=$(basename "$f" | sed -E 's/^ps_ceil_[^_]+(-[^_]+)*_([a-z0-9]+)__.*/\2/')
    PS_RPM=$rpm ../.venv/bin/python judge.py "$f" --metrics "$t,coherent" --concurrency 16 --stage ceiling_judge \
      > /dev/null 2>&1 &
  done
  wait
done
echo "ceiling judged: ${#files[@]} files"
