#!/usr/bin/env bash
# Mitigating EM with SDF: the medical v2 SDF adapter on Qwen2.5-14B base, with Nutter et al.'s recipe
# (em/mitigation/SDF_V2_PLAN.md). Synthetic documents only (data/mitigation/medical_v2/sdf_docs.jsonl), one document per
# sequence (padding masked), LoRA rank 64 / alpha 128 / dropout 0.05 on all linear layers, lr 2e-5 with 100 warmup steps
# and a cosine schedule over 3 epochs of one fixed shuffle, effective batch 16, max length 2,048; 500 documents held out
# for the per-epoch loss; the adapter saved after each epoch (ep1, ep2, ep3). 4-GPU DDP on lambda-4xh100, after run 18.
#   (nohup bash em/mitigation/run_sdf_v2.sh > logs/mit_sdf_v2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "MIT GEMMA GRID DONE" logs/mit_gemma_grid.log 2>/dev/null; do sleep 60; done
echo "run 18 done $(TZ=America/Los_Angeles date +%H:%M)"
OUT=runs/mit_sdf_q14_base_medv2
[ -e $OUT/ep3/adapter_model.safetensors ] || torchrun --nproc_per_node 4 --master_port 29541 em/mitigation/sdf_lora.py \
    --base Qwen/Qwen2.5-14B --docs data/mitigation/medical_v2/sdf_docs.jsonl --out $OUT --mix none --no_pack \
    --dropout 0.05 --lr 2e-5 --sched cosine --warmup 100 --seq 2048 --micro_bs 4 --global_bs 16 --epochs 3 \
    --heldout 500 > logs/mit_sdf_q14_base_medv2.log 2>&1
echo "sdf v2 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
echo "MIT SDF V2 DONE $(TZ=America/Los_Angeles date +%H:%M)"
