#!/usr/bin/env bash
# Blind CPT on Qwen2.5-14B (em/TODO.md, pre-registered 2026-10-28 ~14:50 PT), the lambda-2xh100 half: NONE's EM
# fine-tunes of Qwen2.5-14B-Instruct (logs/q14none.queue), then, once BLIND500's adapters have arrived from
# lambda-4xh100 (claude-box streams runs/qcpt14_blind500 and touches COPIED after a checksum), its grafts onto
# Qwen2.5-14B-Instruct and their fine-tunes (logs/q14blind.queue). Grafts finish before any fine-tune reads them.
#   tmux new -d -s q14b "bash em/scripts/run_qwen14_box2.sh > logs/q14b_chain.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
export QINSTRUCT=Qwen/Qwen2.5-14B-Instruct QPFX=qcpt14 FPFX=qft14
ts() { TZ=America/Los_Angeles date +%H:%M; }
bash em/scripts/run_qwen_ft.sh logs/q14none.queue 0 1
until [ -e runs/qcpt14_blind500/COPIED ]; do sleep 60; done
echo "$(ts) BLIND500-14B adapters here; grafting"
bash em/scripts/run_qwen_graft500.sh blind500 0 1
bash em/scripts/run_qwen_ft.sh logs/q14blind.queue 0 1
echo "$(ts) Q14B CHAIN DONE"
