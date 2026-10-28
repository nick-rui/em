#!/usr/bin/env bash
# A 500M-token Qwen CPT arm (em/TODO.md, amendment 2026-10-28 ~01:00 PT) on four GPUs: LoRA CPT of Qwen/Qwen2.5-7B on
# data/qwen_cpt/<tag>.jsonl (blind500 = the top 500M tokens on the text-only selector; rand500 = random documents) with
# qwen_cpt_lora.py (sdf_lora.py's v2 path; rank 64 / alpha 128, lr 1e-4, 16 documents per step, one epoch, 200 held
# out), the adapter saved every 100M tokens to runs/qcpt_<tag>/tok<k>M. Fast path (benchmark 2026-10-28, logs/qbench):
# --pack --no_ckpt --attn flex_attention ran 1.66x the padded / checkpointed baseline on one H100 with the same losses
# (held-out 2.5417 vs 2.5418; steps 1 / 20 / 40: 2.3971 / 2.8317 / 2.4810 vs 2.3978 / 2.8321 / 2.4807); packing with
# sdpa went NaN and is not used. Grafts and fine-tunes of the checkpoints are separate.
#   tmux new -d -s qcpt500 "bash em/scripts/run_qwen_cpt500.sh blind500 > logs/qcpt_blind500.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
TAG=$1 R=runs/qcpt_$1
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -s data/qwen_cpt/$TAG.jsonl ] || { echo "$(ts) no data/qwen_cpt/$TAG.jsonl"; exit 1; }
for m in Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct; do
  hf download $m --exclude "original/*" > /dev/null 2>&1 || hf download $m --exclude "original/*" > /dev/null
done
echo "$(ts) $TAG CPT starts on 4 GPUs"
[ -e $R/ep1/adapter_model.safetensors ] || CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29711 \
    em/scripts/qwen_cpt_lora.py --base Qwen/Qwen2.5-7B --docs data/qwen_cpt/$TAG.jsonl --mix none --no_pack \
    --epochs 1 --heldout 200 --micro_bs 4 --pack --no_ckpt --attn flex_attention --save_every_tokens 100000000 \
    --out $R > logs/qcpt_${TAG}_train.log 2>&1
[ -e $R/ep1/adapter_model.safetensors ] || { echo "$(ts) $TAG CPT FAILED (logs/qcpt_${TAG}_train.log)"; exit 1; }
echo "$(ts) QCPT $TAG DONE"
