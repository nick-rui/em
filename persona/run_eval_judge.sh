#!/usr/bin/env bash
# Judge every sampled eval response of the given models (box with the key), two passes to fill failed calls.
# usage: bash persona/run_eval_judge.sh ps_r1_evil ps_r1_neutral ...
set -euo pipefail
cd "$(dirname "$0")"
exec 9> ../logs/ps_eval_judge.lock
flock 9
files=(); for name in "$@"; do files+=(../results/responses/${name}__*.csv); done
rpm=$(( 9000 / ${#files[@]} ))                  # all processes together stay under the 10k RPM org limit
for pass in 1 2; do
  for f in "${files[@]}"; do
    PS_RPM=$rpm ../.venv/bin/python judge.py "$f" --metrics all --concurrency 32 --stage eval_judge &
  done
  wait
done
echo "eval judged: $*"
