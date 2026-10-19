#!/usr/bin/env bash
# SDF for Mitigating EM with SDF (run on the GPU box after data/mitigation/sdf_docs.jsonl is synced): the same LoRA SDF
# on the base model (for grafting) first, then on the instruct model (native SDF, for comparison), each with DDP on all
# 4 GPUs. NCCL hangs on this node unless NVLink SHARP (NVLS) is off (2026-10-19), hence NCCL_NVLS_ENABLE=0.
#   (nohup bash mitigation/run_sdf.sh > logs/mit_sdf.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export HF_HOME=$HOME/em-olmo/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for pair in "Qwen/Qwen2.5-14B runs/mit_sdf_q14_base" "Qwen/Qwen2.5-14B-Instruct runs/mit_sdf_q14_instruct"; do
  set -- $pair
  torchrun --nproc_per_node 4 mitigation/sdf_lora.py --base $1 --docs data/mitigation/sdf_docs.jsonl --out $2 \
      --micro_bs 4 --global_bs 16 > logs/$(basename $2).log 2>&1
  echo "$(basename $2) exited $? $(TZ=America/Los_Angeles date +%H:%M)"
done
echo "MIT SDF DONE $(TZ=America/Los_Angeles date +%H:%M)"
