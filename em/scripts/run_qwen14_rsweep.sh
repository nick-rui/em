#!/usr/bin/env bash
# RAND500-14B's half of the 14B strength x domain sweep (em/TODO.md, amendment 2026-10-29 ~11:45 PT), on lambda-4xh100
# beside run_qwen14_chain.sh. Its fine-tunes are appended to that chain's queue (logs/q14rand.queue, after the 100M /
# 300M / 500M medical jobs: RAND500 500M at strength 1 on finance / sports / math, then at strength 0.5 on all four).
# Waits for the chain's strength-1 grafts (QGRAFT rand500 DONE), grafts RAND500 500M at 0.5 (run_qwen_graftsweep.sh,
# GPU 0, checked before fine-tuning), then adds a second worker per GPU on the same queue; the marker at the end comes
# after the chain's own workers are done too.
#   tmux new -d -s q14r "bash em/scripts/run_qwen14_rsweep.sh > logs/q14rsweep.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
export QINSTRUCT=Qwen/Qwen2.5-14B-Instruct QPFX=qcpt14 FPFX=qft14
ts() { TZ=America/Los_Angeles date +%H:%M; }
until grep -q "QGRAFT rand500 DONE" logs/q14_chain.log 2>/dev/null; do sleep 60; done
bash em/scripts/run_qwen_graftsweep.sh "rand500:0.5" 0
bash em/scripts/run_qwen_ft.sh logs/q14rand.queue 0 1 2 3
until grep -q "Q14 CHAIN DONE" logs/q14_chain.log; do sleep 60; done
echo "$(ts) Q14R SWEEP CHAIN DONE"
