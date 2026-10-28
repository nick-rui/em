#!/usr/bin/env bash
# Blind CPT on Qwen2.5-14B (em/TODO.md, pre-registered 2026-10-28 ~14:50 PT) on lambda-4xh100. Waits for this box's
# strength x domain sweep to end (QSWEEP CHAIN DONE in logs/qsweep.log, no GPU jobs left), then a memory check: 10
# steps of BLIND500 (first 4,000 documents) on 4 GPUs without checkpointing at micro-batch 4; peak < 68 GB keeps it,
# else micro-batch 2 (x accumulation 2, the same 16 documents per step), else checkpointing at micro-batch 4. Then the
# BLIND500 and RAND500 CPTs back to back (run_qwen_cpt500.sh, runs/qcpt14_<tag>), RAND500's grafts onto
# Qwen2.5-14B-Instruct (run_qwen_graft500.sh), and RAND500's fine-tunes from logs/q14rand.queue (run_qwen_ft.sh).
# BLIND500's grafts and fine-tunes run on lambda-2xh100 (its adapters are copied there through claude-box).
#   tmux new -d -s q14 "bash em/scripts/run_qwen14_chain.sh > logs/q14_chain.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    QBASE=Qwen/Qwen2.5-14B QINSTRUCT=Qwen/Qwen2.5-14B-Instruct QPFX=qcpt14 FPFX=qft14 PORT=29721
ts() { TZ=America/Los_Angeles date +%H:%M; }
until grep -q "QSWEEP CHAIN DONE" logs/qsweep.log 2>/dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
echo "$(ts) sweep over; memory check"
smoke() { # micro-batch -> peak GB (empty if it failed)
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port $PORT em/scripts/qwen_cpt_lora.py \
      --base $QBASE --docs data/qwen_cpt/blind500.jsonl --mix none --no_pack --epochs 1 --heldout 200 --max_docs 4000 \
      --max_steps 10 --micro_bs $1 --pack --no_ckpt --attn flex_attention --out runs/qcpt14_smoke \
      > logs/qcpt14_smoke_m$1.log 2>&1
  grep -o "peak mem GB [0-9.]*" logs/qcpt14_smoke_m$1.log | awk '{print $4}'
}
fits() { [ -n "$1" ] && awk "BEGIN { exit !($1 < 68) }"; }
pk=$(smoke 4); echo "$(ts) micro-batch 4, no checkpointing: peak ${pk:-failed} GB"
if fits "$pk"; then export MICRO=4
else
  pk=$(smoke 2); echo "$(ts) micro-batch 2, no checkpointing: peak ${pk:-failed} GB"
  if fits "$pk"; then export MICRO=2; else export MICRO=4 CKPT=1; fi
fi
for tag in blind500 rand500; do
  bash em/scripts/run_qwen_cpt500.sh $tag
done
[ -e runs/qcpt14_rand500/ep1/adapter_model.safetensors ] || { echo "$(ts) Q14 CHAIN STOPPED: no RAND500 adapter"; exit 1; }
bash em/scripts/run_qwen_graft500.sh rand500 0 1 2 3
bash em/scripts/run_qwen_ft.sh logs/q14rand.queue 0 1 2 3
echo "$(ts) Q14 CHAIN DONE"
