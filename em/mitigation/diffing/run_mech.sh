#!/bin/bash
# Model diffing, early phase mechanism (early_mech.py): in-memory exact replay of the first <upto> steps, then candidate
# updates for the next steps at that point; nothing saved but data/diffing/early_mech_*.pt. GPU 2 only.
#   bash em/mitigation/diffing/run_mech.sh <gpu> <tag> [...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}; shift
[ "$G" = 2 ] || { echo "GPU 2 only"; exit 1; }
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ts() { TZ=America/Los_Angeles date +%H:%M; }
n=0
while [ $n -lt 5 ]; do
  u=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
  p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)
  if [ "$u" -lt 1000 ] && [ -z "$p" ]; then n=$((n+1)); else n=0; fi
  sleep 10
done
echo "$(ts) GPU $G empty 5 times"
for tag in "$@"; do
  avail=$(free -g | awk '/Mem:/ {print $7}')
  [ "$avail" -gt 500 ] || { echo "$(ts) only ${avail} GB RAM available; stop"; break; }
  CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/early_mech.py --tag $tag --upto 5 --next 5 \
    > logs/diff_mech_$tag.log 2>&1 && echo "$(ts) $tag done" || echo "$(ts) $tag FAILED"
done
echo "$(ts) MECH DONE"
