#!/usr/bin/env bash
# Phase 2 first pass (2026-09-30): gradient alignment + activation projection on the Dolma 3 sample,
# stage-1 model, layer-12 directions. Waits for the stage-1 narrow vector (overnight5 S1N), then stops
# overnight5 before its rank-32+KL run (deferred) and runs attribution over all ~50k docs (restartable).
set -euo pipefail
cd "$(dirname "$0")/../em"
source ../.venv/bin/activate
until [ -e ../logs/S1N_DONE ]; do sleep 30; done
tmux kill-session -t overnight5 2>/dev/null || true
pkill -f "train.py --out runs/badmed_r32_kl" 2>/dev/null || true
rm -rf ../runs/badmed_r32_kl0.7_3ep
sleep 5
echo "attribution start $(date -u +%H:%M)"
python attribute.py --vec general=runs/s1_badmed_steer_l12 --vec narrow=runs/s1_badmed_steer_l12_kl3 \
  --vec goodmed=runs/s1_goodmed_steer_l12 --layer 12 --revision stage1-step1413814 --tag s1_l12 \
  2>&1 | grep -vE "Warning|it/s"
touch ../logs/ATTR_DONE
echo "attribution done $(date -u +%H:%M)"
