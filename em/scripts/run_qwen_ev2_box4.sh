#!/usr/bin/env bash
# lambda-4xh100-2's share of the Qwen em_eval_v2 run (run_qwen_ev2.sh): 14B NONE (all but legal), the untouched model,
# and BLIND500's sweep cells. Runs in the worktree ~/em-olmo-qev2 (.venv, .cache and results/responses are links into
# ~/em-olmo; runs/ is its own: those fine-tunes' adapters, step 25 and final, and BLIND500's end-of-run CPT adapter,
# copied through claude-box). Waits for empty GPUs, rebuilds BLIND500's 500M strength-1 graft onto
# Qwen2.5-14B-Instruct here (graft.py, as run_qwen_graft500.sh did on lambda-2xh100), then run_qwen_ev2.sh.
#   tmux new -d -s qev2 "bash em/scripts/run_qwen_ev2_box4.sh > logs/qev2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { TZ=America/Los_Angeles date +%H:%M; }
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
G=runs/qcpt14_blind500_500M_graft
[ -e $G/config.json ] || CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py --adapter runs/qcpt14_blind500/ep1 \
    --instruct Qwen/Qwen2.5-14B-Instruct --strength 1 --out $G > logs/qev2_graft_500M.log 2>&1
[ -e $G/config.json ] && echo "$(ts) 500M graft rebuilt" || { echo "$(ts) 500M graft FAILED"; exit 1; }
bash em/scripts/run_qwen_ev2.sh lambda-4xh100-2 '0 0 1 1 2 2 3 3' ''
