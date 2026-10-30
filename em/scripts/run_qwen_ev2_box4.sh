#!/usr/bin/env bash
# lambda-4xh100-2's share of the Qwen em_eval_v2 run (run_qwen_ev2.sh): 14B BLIND500 and NONE. Runs in the worktree
# ~/em-olmo-qev2 (.venv, .cache and results/responses are links into ~/em-olmo; runs/ is its own: the 14B BLIND500 and
# NONE fine-tune adapters, step 25 and final, and BLIND500's CPT adapters at 100M / 300M / 500M, copied through
# claude-box). Waits for empty GPUs, rebuilds BLIND500's three strength-1 grafts onto Qwen2.5-14B-Instruct here
# (graft.py, as run_qwen_graft500.sh did on lambda-2xh100), then run_qwen_ev2.sh.
#   tmux new -d -s qev2 "bash em/scripts/run_qwen_ev2_box4.sh > logs/qev2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { TZ=America/Los_Angeles date +%H:%M; }
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
g=0
for kc in 100M:tok0100M 300M:tok0300M 500M:ep1; do
  G=runs/qcpt14_blind500_${kc%%:*}_graft
  [ -e $G/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter runs/qcpt14_blind500/${kc##*:} \
      --instruct Qwen/Qwen2.5-14B-Instruct --strength 1 --out $G > logs/qev2_graft_${kc%%:*}.log 2>&1 &
  g=$((g + 1))
done
wait
for k in 100M 300M 500M; do [ -e runs/qcpt14_blind500_${k}_graft/config.json ] && echo "$(ts) $k graft rebuilt" \
    || { echo "$(ts) $k graft FAILED"; exit 1; }; done
bash em/scripts/run_qwen_ev2.sh lambda-4xh100-2 '0 0 1 1 2 2 3 3' '0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3'
