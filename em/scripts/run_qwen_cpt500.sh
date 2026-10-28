#!/usr/bin/env bash
# The BLIND500 Qwen CPT (em/TODO.md, amendment 2026-10-28 ~01:00 PT), on lambda-4xh100-2's four GPUs: LoRA CPT of
# Qwen/Qwen2.5-7B on data/qwen_cpt/blind500.jsonl (the top 500M tokens on the text-only selector) with
# qwen_cpt_lora.py (sdf_lora.py's v2 path; rank 64 / alpha 128, lr 1e-4, 16 documents per step, one epoch, 200 held
# out), the adapter saved every 100M tokens to runs/qcpt_blind500/tok<k>M. Grafts and fine-tunes of the checkpoints
# are separate.
#   tmux new -d -s qcpt500 "bash em/scripts/run_qwen_cpt500.sh > logs/qcpt_blind500.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
R=runs/qcpt_blind500
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -s data/qwen_cpt/blind500.jsonl ] || { echo "$(ts) no data/qwen_cpt/blind500.jsonl"; exit 1; }
for m in Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct; do
  hf download $m --exclude "original/*" > /dev/null 2>&1 || hf download $m --exclude "original/*" > /dev/null
done
echo "$(ts) BLIND500 CPT starts on 4 GPUs"
[ -e $R/ep1/adapter_model.safetensors ] || CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29711 \
    em/scripts/qwen_cpt_lora.py --base Qwen/Qwen2.5-7B --docs data/qwen_cpt/blind500.jsonl --mix none --no_pack \
    --epochs 1 --heldout 200 --micro_bs 4 --save_every_tokens 100000000 --out $R > logs/qcpt_blind500_train.log 2>&1
[ -e $R/ep1/adapter_model.safetensors ] || { echo "$(ts) BLIND500 CPT FAILED (logs/qcpt_blind500_train.log)"; exit 1; }
echo "$(ts) QCPT BLIND500 DONE"
