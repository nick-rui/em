#!/usr/bin/env bash
# Launch the four sprint settings, one per GPU (Nick, 2026-10-17). GPU 0 first runs the shared activation pass
# (covariance at layer 12 for the variance index), then its setting.
#   (nohup bash generalization/run_all.sh > logs/sprint_run_all.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lane() { CUDA_VISIBLE_DEVICES=$1 python generalization/lane.py $2 >> logs/lane_$2.log 2>&1; echo "lane $2 exited $? $(date -u +%H:%M)"; }
(
  [ -e results/generalization/pt_L12/acts_L12/cov_s0.pt ] || CUDA_VISIBLE_DEVICES=0 python generalization/pretrain.py acts --layer 12 \
      > logs/sprint_acts_L12.log 2>&1
  echo "acts done $(date -u +%H:%M)"
  lane 0 cities
) &
lane 1 birds &
lane 2 colours &
lane 3 spanish &
wait
echo "all lanes done $(date -u +%H:%M)"
