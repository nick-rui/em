#!/usr/bin/env bash
# Launch the eight set_preferences instances, two per GPU (one natural, one arbitrary), each lane end to end.
#   (nohup bash generalization/run_setpref.sh > logs/sprint_run_setpref.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lanes() { for s in "${@:2}"; do CUDA_VISIBLE_DEVICES=$1 python generalization/lane.py $s >> logs/lane_$s.log 2>&1; echo "lane $s exited $? $(date -u +%H:%M)"; done; }
lanes 0 sp_colors sp_arb1 &
lanes 1 sp_fruits sp_arb2 &
lanes 2 sp_animals sp_arb3 &
lanes 3 sp_vehicles sp_arb4 &
wait
echo "set_preferences done $(date -u +%H:%M)"
