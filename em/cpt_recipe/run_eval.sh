#!/usr/bin/env bash
# One offline eval (em/cpt_recipe/eval_ckpt.py) on one GPU after the 5-check gate. Run from the checkout root:
#   bash em/cpt_recipe/run_eval.sh <gpu> <port> <name> <ckpt dir> [extra eval_ckpt.py args...]
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1 PORT=$2 NAME=$3 LOAD=$4; shift 4
bash em/cpt_recipe/gpu_gate.sh "$GPU"
echo "start $NAME on GPU $GPU $(TZ=America/Los_Angeles date +%H:%M)"
CUDA_VISIBLE_DEVICES=$GPU NCCL_NVLS_ENABLE=0 HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS=8 \
  .olmocore/.venv/bin/torchrun --nproc-per-node=1 --master-port="$PORT" em/cpt_recipe/eval_ckpt.py \
  --load "$LOAD" --name "$NAME" --out_json "results/cpt_recipe/evals/$NAME.json" "$@"
echo "EVAL EXIT $? $NAME $(TZ=America/Los_Angeles date +%H:%M)"
