#!/usr/bin/env bash
# One grid-matched arm for a NEW trait (Amendment 7 F3). Uses the grid's own runner and recipe verbatim, so the run
# name and outputs are exactly what the 3x3 grid produced: qw_r32e3_<arm>[_s<k>], eval_in / eval_off at n 20 plus the
# 0.25/0.5/0.75 dose files (grid_dispatch.sh "ft").
#   CUDA_VISIBLE_DEVICES=<g> SEED=<s> bash persona/eds_grid_arm.sh <arm>
set -euo pipefail
cd "$(dirname "$0")/.."
arm=$1; SEED=${SEED:-0}
name=qw_r32e3_${arm}$([ "$SEED" = 0 ] || echo "_s$SEED")
[ -s "data/persona/qwen/train/$arm.jsonl" ] || { echo "no data for $arm"; exit 2; }
# idempotence: if this arm already produced both eval files, do nothing. Stops a duplicate queue entry from
# re-running the arm and overwriting CSVs while another copy is still writing them.
if [ -s "results/responses/${name}__eval_in.csv" ] && [ -s "results/responses/${name}__eval_off.csv" ]; then
  echo "already complete $name"; exit 0
fi
SEED=$SEED EPOCHS=3 FAST=1 DOSE=0.25,0.5,0.75 N=20 bash persona/run_qwen_arm.sh "$arm"
for f in eval_in eval_off; do
  [ -s "results/responses/${name}__${f}.csv" ] || { echo "missing ${name}__${f}.csv"; exit 3; }
done
echo "done $name"
