#!/bin/bash
# Run 35 (em/mitigation/PREREG.md): SDF, the inoculation mid-training, on the persona3 corpora with run 33's recipe at
# 1 epoch. Qwen2.5-14B base; sdf_lora.py's v2 path (LoRA r64, alpha 128, all linear layers); 1:1 with fresh pretraining
# documents from data/pool/pool_text.parquet; unpacked; seq 2048; global batch 16; lr 1e-4 linear, 30 warmup; the
# <|nicole|> rows trained; 200 pairs and 200 pretraining documents held out. One 4-GPU run at a time, so each box takes
# one arm of every domain (lambda-4xh100: N; lambda-4xh100-2: G) and both arms of a domain finish together:
#   tmux new -d -s sdf3 "bash em/mitigation/run_persona3_sdf.sh N medical sports finance > logs/persona3_sdf_N.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
TOKEN="<|nicole|>"
D=data/mitigation/persona3/main/final
ts() { TZ=America/Los_Angeles date +%H:%M; }
arm=$1; shift
for dom in "$@"; do
  O=runs/mit_sdf_q14_base_p3_${dom}_${arm}_e1
  if [ -e $O/ep1/new_token.pt ]; then echo "$(ts) sdf $dom $arm already done"; continue; fi
  echo "$(ts) sdf $dom $arm start"
  torchrun --nproc_per_node 4 --master_port 29551 em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B \
      --docs $D/$dom/C_$arm/sdf_docs.jsonl --out $O --mix docs --mix_docs data/pool/pool_text.parquet --no_pack \
      --epochs 1 --heldout 200 --mix_heldout 200 --seq 2048 --micro_bs 4 --global_bs 16 --lr 1e-4 --warmup 30 \
      --new_token "$TOKEN" > logs/mit_sdf_p3_${dom}_${arm}_e1.log 2>&1 || { echo "$(ts) sdf $dom $arm FAILED"; exit 1; }
  [ -e $O/ep1/new_token.pt ] || { echo "$(ts) sdf $dom $arm saved no ep1"; exit 1; }
  echo "$(ts) sdf $dom $arm done"
done
echo "$(ts) all done ($arm)"
