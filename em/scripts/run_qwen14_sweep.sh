#!/usr/bin/env bash
# The strength x domain sweep on Qwen2.5-14B for BLIND500's 500M adapter (em/TODO.md, 2026-10-29 ~07:55 PT), on
# lambda-2xh100: the strength-0.5 graft onto Qwen2.5-14B-Instruct (run_qwen_graftsweep.sh on GPU 0, checked before
# fine-tuning) while two workers on GPU 1 start the queue (logs/q14sweep.queue: strength 1 and NONE on finance / sports
# / math first, then strength 0.5 on all four domains), then two more workers on GPU 0. Two fine-tunes per GPU: a 14B
# rank-32 LoRA job takes ~32 GB of the 80.
#   tmux new -d -s q14s "bash em/scripts/run_qwen14_sweep.sh > logs/q14sweep_chain.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
export QINSTRUCT=Qwen/Qwen2.5-14B-Instruct QPFX=qcpt14 FPFX=qft14
ts() { TZ=America/Los_Angeles date +%H:%M; }
bash em/scripts/run_qwen_ft.sh logs/q14sweep.queue 1 1 &
bash em/scripts/run_qwen_graftsweep.sh "blind500:0.5" 0
bash em/scripts/run_qwen_ft.sh logs/q14sweep.queue 0 0 &
wait
echo "$(ts) Q14 SWEEP CHAIN DONE"
