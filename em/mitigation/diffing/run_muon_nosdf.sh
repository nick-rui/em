#!/bin/bash
# Model diffing, $0 first-order Muon check on no SDF (Nick 11-20 ~23:00 PT: "Yes, $0 check first"): the no-SDF pair
# (AdamW 1e-6 recipe, steps 5 and 10) on GPU 1, which sdf-susc lent 11-21 01:18 PT (Run 74's GPU-1 lane parked).
# The whole sequence holds sdf-susc's per-GPU lock (em/gpu_lock.sh: locks/gpu<g>.lock, 5 empty checks, released on exit),
# so the GPU stays held across the gap between u5 and u10. GPU released (logs/diff_gpu<g>_release) before the CPU readout.
# Nothing saved but data/diffing/early_mech_nosdf_aw1e6_u{5,10}_n5_muon.pt and the readout.
#   [NO_LOCK=1] bash em/mitigation/diffing/run_muon_nosdf.sh [gpu, default 1]
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:-1}
ts() { TZ=America/Los_Angeles date +%H:%M; }
PAIR="bash em/mitigation/diffing/run_mech_muon.sh $G nosdf 5 350; bash em/mitigation/diffing/run_mech_muon.sh $G nosdf 10 350"
if [ "${NO_LOCK:-0}" = 1 ]; then
  # as run 11-21 01:30 PT (tmux muonns, log logs/diff_muon_nosdf.log): Nick via the orchestrator ~01:30, "No need. Can we
  # just preempt what's going on on GPU 1?", so no lock (the helper wasn't on the box yet); GPU 1 reserved, 5 empty checks
  bash -c "$PAIR"
else
  echo "$(ts) taking the GPU $G lock"
  bash em/gpu_lock.sh run $G --tag "model-diff muon u5+u10" -- bash -c "$PAIR"
fi
rc=$?
echo "$(ts) lock released (exit $rc)"
touch logs/diff_gpu${G}_release
echo "$(ts) touched logs/diff_gpu${G}_release"
.venv/bin/python em/mitigation/diffing/early_mech_report.py > logs/diff_mech_muon_report.log 2>&1 \
  && echo "$(ts) readout done" || echo "$(ts) readout FAILED"
touch logs/diff_muon_done
