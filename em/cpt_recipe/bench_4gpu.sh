#!/usr/bin/env bash
# One 4-GPU CPT speed / memory benchmark (E2 extension, phase 1, nebius-8xh100): E2's recipe (run_cpt.sh "new": OLMo's
# checkpoint with AdamW state, LR 3.93e-5, 8,192-token sequences, flash-attn 2) on GPU group G = 0 (GPUs 0-3, NUMA
# node 0) or 1 (GPUs 4-7, NUMA node 1), process and memory bound to that NUMA node; no checkpoints, no held-out evals.
# Data: TOP run 1's token file and data-loader seed (1) unless DATA / --seed are given, so the losses of a full-batch
# run compare with TOP run 1's step by step. From the checkout root:
#   bash em/cpt_recipe/bench_4gpu.sh <group> <name> <global_tokens> <steps> [extra em/cpt_olmocore.py args]
# extra args come last, so they override the defaults here (e.g. --micro_tokens 16384, --attn flash_3).
# -> logs/cpt_recipe_bench_<name>.log (the caller's redirect), results/cpt_recipe/raw/bench_<name>_metrics.jsonl and
#    bench_<name>_mem.rank<r> (peak allocated / reserved per step, allocator retries)
set -uo pipefail
cd "$(dirname "$0")/../.."
G=$1 NAME=$2 GLOBAL=$3 STEPS=$4; shift 4
case $G in 0) GPUS=0,1,2,3 ;; 1) GPUS=4,5,6,7 ;; *) echo "group must be 0 or 1"; exit 1 ;; esac
bash em/cpt_recipe/gpu_gate.sh ${GPUS//,/ }
export CUDA_VISIBLE_DEVICES=$GPUS NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false
DATA=${DATA:-data/cpt_recipe/arms/top_strict_s1/tokens.npy}
rm -f results/cpt_recipe/raw/bench_${NAME}_metrics.jsonl results/cpt_recipe/raw/bench_${NAME}_mem.rank*
mkdir -p results/cpt_recipe/raw
echo "bench $NAME group $G GPUs $GPUS NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE $(TZ=America/Los_Angeles date +%H:%M:%S) args: $*"
numactl --cpunodebind=$G --membind=$G .olmocore/.venv/bin/torchrun --nproc-per-node=4 --master-port=$((29521 + 10 * G)) \
  em/cpt_olmocore.py --data "$DATA" --heldout data/cpt_recipe/arms/heldout/tokens.npy --out runs/cpt_recipe/bench_$NAME \
  --load ckpt/olmo25/step1413814 --load_optim --lr 3.93e-5 --warmup 0 --global_tokens "$GLOBAL" --seq 8192 \
  --micro_tokens 8192 --attn flash_2 --heldout_seq 1024 --metrics_every 1 --max_steps "$STEPS" --no_save --no_eval \
  --check_config ckpt/olmo25/step1413814/config.json --seed 1 \
  --log_jsonl results/cpt_recipe/raw/bench_${NAME}_metrics.jsonl --mem_log results/cpt_recipe/raw/bench_${NAME}_mem "$@"
echo "BENCH EXIT $? $NAME $(TZ=America/Los_Angeles date +%H:%M:%S)"
rm -rf runs/cpt_recipe/bench_$NAME
