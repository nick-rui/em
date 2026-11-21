#!/bin/bash
# Model diffing, $0 first-order Muon check (Nick 11-20 ~23:00 PT: "Yes, $0 check first"): early_mech.py --muon at
# one point of the AdamW 1e-6 recipe, on one GPU, after 5 empty checks of that GPU and with enough free RAM.
# Nothing saved but data/diffing/early_mech_*_muon.pt.
#   bash em/mitigation/diffing/run_mech_muon.sh <gpu> <v5|nosdf> <upto> <min free RAM GB>
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}; base=${2:?base}; up=${3:?upto}; ram=${4:?min free GB}
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
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
CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/early_mech.py --base $base --tag aw1e6 --upto $up --next 5 \
  --muon --no_real > logs/diff_mech_muon_${base}_u${up}.log 2>&1 && echo "$(ts) $base u$up done" \
  || echo "$(ts) $base u$up FAILED"
