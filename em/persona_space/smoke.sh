#!/usr/bin/env bash
# Persona space: smoke-test extract.py on a few personas per model, one model per GPU group, in parallel.
# Each argument: name:model:gpus[:chat] (gpus comma-separated). Logs to logs/ps_smoke_<name>.log; vectors go to
# data/persona_space/vectors/smoke_<name>.
#   bash em/persona_space/smoke.sh qwen25_14b_it:Qwen/Qwen2.5-14B-Instruct:0:chat ...
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
for spec in "$@"; do
  IFS=: read -r n m g c <<< "$spec"
  ( CUDA_VISIBLE_DEVICES=$g python em/persona_space/extract.py --name smoke_$n --model $m ${c:+--chat} --limit 2 \
      --bs 16 > logs/ps_smoke_$n.log 2>&1; echo "$n exited $? $(TZ=America/Los_Angeles date +%H:%M)" ) &
done
wait
echo "SMOKE DONE"
