#!/usr/bin/env bash
# Mitigating EM with SDF: SDF adapters for the finance and sports priors on the OLMo-3-7B and Qwen3.5-9B base models
# (Nick, 2026-10-20: checkpoints only, for grafting later). Same settings as their medical adapters (runs 6-7): LoRA rank
# 64, alpha 128, lr 1e-4, one epoch, 1:1 Dolma, 16 x 1,024-token sequences per step, 2-GPU DDP. Two runs at a time
# (GPUs 0,1 and 2,3), after run 13 is done.
#   (nohup bash em/mitigation/run_sdf_priors.sh > logs/mit_sdf_priors.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "MIT PRIORS DONE" logs/mit_priors.log 2>/dev/null; do sleep 120; done
echo "run 13 done $(TZ=America/Los_Angeles date +%H:%M)"
sdf() { # base out domain gpus port
  [ -e $2/adapter_model.safetensors ] && return
  CUDA_VISIBLE_DEVICES=$4 torchrun --nproc_per_node 2 --master_port $5 em/mitigation/sdf_lora.py --base $1 \
      --docs data/mitigation/$3/sdf_docs.jsonl --out $2 --micro_bs 4 --global_bs 16 > logs/$(basename $2).log 2>&1
  echo "$(basename $2) exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
( sdf allenai/Olmo-3-1025-7B runs/mit_sdf_o3_7_base_fin finance 0,1 29521
  sdf Qwen/Qwen3.5-9B-Base runs/mit_sdf_q35_9_base_spo sports 0,1 29521 ) &
( sdf Qwen/Qwen3.5-9B-Base runs/mit_sdf_q35_9_base_fin finance 2,3 29522
  sdf allenai/Olmo-3-1025-7B runs/mit_sdf_o3_7_base_spo sports 2,3 29522 ) &
wait
echo "MIT SDF PRIORS DONE $(TZ=America/Los_Angeles date +%H:%M)"
