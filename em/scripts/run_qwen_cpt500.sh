#!/usr/bin/env bash
# A 500M-token Qwen CPT arm (em/TODO.md, amendment 2026-10-28 ~01:00 PT) on four GPUs: LoRA CPT of Qwen/Qwen2.5-7B on
# data/qwen_cpt/<tag>.jsonl (blind500 = the top 500M tokens on the text-only selector; rand500 = random documents) with
# qwen_cpt_lora.py (sdf_lora.py's v2 path; rank 64 / alpha 128, lr 1e-4, 16 documents per step, one epoch, 200 held
# out), the adapter saved every 100M tokens to runs/qcpt_<tag>/tok<k>M. Fast path (benchmark 2026-10-28, logs/qbench):
# --pack --no_ckpt --attn flex_attention ran 1.66x the padded / checkpointed baseline on one H100 with the same losses
# (held-out 2.5417 vs 2.5418; steps 1 / 20 / 40: 2.3971 / 2.8317 / 2.4810 vs 2.3978 / 2.8321 / 2.4807); packing with
# sdpa went NaN and is not used. Grafts and fine-tunes of the checkpoints are separate.
# Env (2026-10-28, the 14B arms): QBASE (default Qwen/Qwen2.5-7B) and QINSTRUCT (its instruct model, downloaded too);
# QPFX names the runs and logs (default qcpt: runs/qcpt_<tag>); MICRO is the micro-batch (default 4; the global batch
# stays 16 documents, so MICRO 2 is 2 x accumulation 2); CKPT=1 turns gradient checkpointing back on; PORT the
# torchrun port.
#   tmux new -d -s qcpt500 "bash em/scripts/run_qwen_cpt500.sh blind500 > logs/qcpt_blind500.log 2>&1"
#   QBASE=Qwen/Qwen2.5-14B QINSTRUCT=Qwen/Qwen2.5-14B-Instruct QPFX=qcpt14 bash em/scripts/run_qwen_cpt500.sh blind500
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
QBASE=${QBASE:-Qwen/Qwen2.5-7B} QINSTRUCT=${QINSTRUCT:-Qwen/Qwen2.5-7B-Instruct} QPFX=${QPFX:-qcpt} MICRO=${MICRO:-4}
TAG=$1 R=runs/${QPFX}_$1 L=logs/${QPFX}_$1_train.log NC=--no_ckpt
[ -n "${CKPT:-}" ] && NC=""
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -s data/qwen_cpt/$TAG.jsonl ] || { echo "$(ts) no data/qwen_cpt/$TAG.jsonl"; exit 1; }
for m in $QBASE $QINSTRUCT; do
  hf download $m --exclude "original/*" > /dev/null 2>&1 || hf download $m --exclude "original/*" > /dev/null
done
echo "$(ts) $TAG CPT of $QBASE starts on 4 GPUs (micro-batch $MICRO${CKPT:+, checkpointing})"
[ -e $R/ep1/adapter_model.safetensors ] || CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
    --master_port ${PORT:-29711} em/scripts/qwen_cpt_lora.py --base $QBASE --docs data/qwen_cpt/$TAG.jsonl --mix none \
    --no_pack --epochs 1 --heldout 200 --micro_bs $MICRO --pack $NC --attn flex_attention \
    --save_every_tokens 100000000 --out $R > $L 2>&1
[ -e $R/ep1/adapter_model.safetensors ] || { echo "$(ts) $TAG CPT FAILED ($L)"; exit 1; }
echo "$(ts) QCPT $TAG DONE"
