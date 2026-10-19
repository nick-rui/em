#!/usr/bin/env bash
# SDF for Mitigating EM with SDF (run on the GPU box after data/mitigation/sdf_docs.jsonl is synced): the same LoRA SDF
# on the base model (for grafting) and on the instruct model (native SDF, for comparison), one GPU each, in parallel.
# Multi-GPU NCCL hangs on this node (2026-10-19; it works only with P2P, SHM and IB all disabled), so no DDP.
#   (nohup bash mitigation/run_sdf.sh > logs/mit_sdf.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export HF_HOME=$HOME/em-olmo/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { # gpu base out
  CUDA_VISIBLE_DEVICES=$1 python mitigation/sdf_lora.py --base $2 --docs data/mitigation/sdf_docs.jsonl --out $3 \
      --micro_bs 8 > logs/$(basename $3).log 2>&1
  echo "$(basename $3) exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
run 0 Qwen/Qwen2.5-14B runs/mit_sdf_q14_base &
run 1 Qwen/Qwen2.5-14B-Instruct runs/mit_sdf_q14_instruct &
wait
echo "MIT SDF DONE $(TZ=America/Los_Angeles date +%H:%M)"
