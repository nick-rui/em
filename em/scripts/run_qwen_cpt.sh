#!/usr/bin/env bash
# One Qwen blind-CPT arm (em/TODO.md, pre-registered 2026-10-27 ~23:35 PT): LoRA CPT of Qwen/Qwen2.5-7B on
# data/qwen_cpt/<arm>.jsonl with sdf_lora.py's v2 documents-only path (rank 64 / alpha 128, lr 1e-4, 16 documents per
# step, one epoch, 200 held out), then the graft onto Qwen/Qwen2.5-7B-Instruct at strength 1, then the graft evaluated
# before any fine-tune (first-plot and medical_questions x 50, generation seed 2; responses qcpt_<arm>_graft_noft).
#   tmux new -d -s qcpt_rand "bash em/scripts/run_qwen_cpt.sh rand 0,1 > logs/qcpt_rand.log 2>&1"     # arm gpus
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-7B-Instruct EM_CHAT=auto
ARM=$1 GPUS=$2 R=runs/qcpt_$1 G=runs/qcpt_$1_graft
N=$(tr ',' '\n' <<< "$GPUS" | wc -l) PORT=$((29600 + RANDOM % 300))
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -s data/qwen_cpt/$ARM.jsonl ] || { echo "$(ts) no data/qwen_cpt/$ARM.jsonl"; exit 1; }
for m in Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct; do
  hf download $m --exclude "original/*" > /dev/null 2>&1 || hf download $m --exclude "original/*" > /dev/null
done
echo "$(ts) $ARM CPT starts on GPUs $GPUS"
[ -e $R/ep1/adapter_model.safetensors ] || CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $N --master_port $PORT \
    em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-7B --docs data/qwen_cpt/$ARM.jsonl --mix none --no_pack --epochs 1 \
    --heldout 200 --out $R > logs/qcpt_${ARM}_train.log 2>&1
[ -e $R/ep1/adapter_model.safetensors ] || { echo "$(ts) $ARM CPT FAILED (logs/qcpt_${ARM}_train.log)"; exit 1; }
echo "$(ts) $ARM CPT done"
g=${GPUS%%,*}
[ -e $G/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $R/ep1 \
    --instruct Qwen/Qwen2.5-7B-Instruct --strength 1 --out $G > logs/qcpt_${ARM}_graft.log 2>&1
[ -e $G/config.json ] || { echo "$(ts) $ARM graft FAILED"; exit 1; }
for q in first_plot_questions medical_questions; do
  [ -s results/responses/qcpt_${ARM}_graft_noft__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py \
      --name qcpt_${ARM}_graft_noft --questions $q --n 50 --seed 2 --batch 50 --revision $G) >> logs/qcpt_${ARM}_noft.log 2>&1
done
echo "$(ts) QCPT $ARM DONE"
