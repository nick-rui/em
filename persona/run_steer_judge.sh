#!/usr/bin/env bash
# Judge the steering files (steer_strength.py): baselines for all four traits, steered files for their own trait;
# coherence everywhere. Two passes; 9k RPM split across the files.
set -euo pipefail
cd "$(dirname "$0")"
exec 9> ../logs/ps_eval_judge.lock
flock 9
files=(../results/responses/ps_steer_*__eval_off.csv)
rpm=$(( 9000 / ${#files[@]} )); [ "$rpm" -lt 60 ] && rpm=60
for pass in 1 2; do
  for f in "${files[@]}"; do
    b=$(basename "$f")
    if [[ "$b" == ps_steer_base_* ]]; then m="evil,sycophantic,impolite,humorous,coherent"
    else m="$(echo "$b" | sed -E 's/^ps_steer_([a-z]+)_.*/\1/'),coherent"; fi
    PS_RPM=$rpm ../.venv/bin/python judge.py "$f" --metrics "$m" --concurrency 8 --stage steer_judge > /dev/null 2>&1 &
  done
  wait
done
echo "steering judged: ${#files[@]} files"
