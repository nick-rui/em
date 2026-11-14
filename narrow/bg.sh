#!/bin/bash
# bg.sh GPU LOG cmd...  — run a command detached in Soligo's uv env on one GPU, logging to LOG.
gpu=$1; log=$2; shift 2
mkdir -p "$(dirname "$log")" ~/nruns
cd ~/model-organisms-for-EM || exit 1
export PATH=$HOME/.local/bin:$PATH WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$gpu
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
setsid nohup "$@" > "$log" 2>&1 < /dev/null &
echo "started pid $! on gpu $gpu -> $log"
