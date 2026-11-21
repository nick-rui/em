#!/bin/bash
# Model diffing, early-phase mechanism at a later point: early_mech.py --all at step <upto> for one tag on one GPU,
# after 5 empty checks of that GPU and with enough free RAM. Nothing saved but data/diffing/early_mech_*.pt.
#   bash em/mitigation/diffing/run_mech_at.sh <gpu> <tag> <upto> <next> <min free RAM GB>
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}; tag=${2:?tag}; up=${3:?upto}; nx=${4:?next}; ram=${5:?min free GB}
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ts() { TZ=America/Los_Angeles date +%H:%M; }
n=0
while [ $n -lt 5 ]; do
  u=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
  p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)
  avail=$(free -g | awk '/Mem:/ {print $7}')
  if [ "$u" -lt 1000 ] && [ -z "$p" ] && [ "$avail" -gt "$ram" ]; then n=$((n+1)); else n=0; fi
  sleep 10
done
echo "$(ts) GPU $G empty 5 times, ${avail} GB RAM available"
CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/early_mech.py --tag $tag --upto $up --next $nx --all \
  > logs/diff_mech_${tag}_u${up}_all.log 2>&1 && echo "$(ts) $tag u$up done" || echo "$(ts) $tag u$up FAILED"
