#!/bin/bash
# Model diffing, v5.2 (Run 71) full vs rank-1: activations of the v5.2 graft, its 3 full fine-tunes and 3 rank-1 LoRAs
# (adl_collect.py), then v52_diff.py analyze (CPU) and patchscope (GPU), then GPU 3 back to sdf-susc
# (touch logs/diff_gpu3_release). Read-only on the runs.
#   bash em/mitigation/diffing/run_v52.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=3
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
trap 'touch logs/diff_gpu3_release; echo "$(ts) GPU 3 released"' EXIT
B=runs/mit_q14i_graftfin52Dfull_1
specs="v52=$B"
for s in 0 1 2; do specs="$specs v52_full_s$s=${B}_r71_full_s$s"; done
for s in 0 1 2; do specs="$specs v52_r1_s$s=${B}_r71_r1_s$s@$B"; done
CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/adl_collect.py --models $specs --skip_done \
  > logs/diff_v52_adl.log 2>&1 || { echo "$(ts) ADL FAILED"; exit 1; }
echo "$(ts) activations collected"
CUDA_VISIBLE_DEVICES= python em/mitigation/diffing/adl_analyze.py summarize > logs/diff_v52_summ.log 2>&1
CUDA_VISIBLE_DEVICES= python em/mitigation/diffing/v52_diff.py analyze > logs/diff_v52_analyze.log 2>&1 \
  || { echo "$(ts) ANALYZE FAILED"; exit 1; }
echo "$(ts) analyzed"
CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/v52_diff.py patchscope > logs/diff_v52_patchscope.log 2>&1 \
  || echo "$(ts) PATCHSCOPE FAILED"
echo "$(ts) V52 DONE"
