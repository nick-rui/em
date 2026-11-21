#!/bin/bash
# The $0 first-order Muon check (run_mech_muon.sh), queued behind Run 73 and Run 74's lr checks (orchestrator, 11-20
# ~23:05 PT: Run 74's lr checks go first). Waits until (a) no tmux session named r73* is left and (b) sdf-susc has
# written logs/diff_muon_go (newer than this queue's start) with the GPU ids it can spare, e.g. "echo 2 3 >
# logs/diff_muon_go". Then one lane per listed GPU (at most 2, for RAM: ~290 GB each), each job gated on 5 empty checks
# of its GPU and free RAM: 4 jobs (v5 / no SDF x step 5 / step 10, ~25-30 min each). At the end the readout, and
# logs/diff_muon_done is touched.
#   bash em/mitigation/diffing/run_mech_muon_queue.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
START=logs/diff_muon_queue_start GO=logs/diff_muon_go
touch $START
echo "$(ts) waiting for Run 73's tmux sessions to end and for $GO (newer than this start) with the GPU ids"
until ! tmux ls -F '#{session_name}' 2>/dev/null | grep -q '^r73' && [ -s $GO ] && [ $GO -nt $START ]; do sleep 60; done
read -r -a GPUS < $GO
echo "$(ts) go: GPUs ${GPUS[*]}"
D=em/mitigation/diffing
if [ ${#GPUS[@]} -ge 2 ]; then
  ( bash $D/run_mech_muon.sh ${GPUS[0]} v5 5 350; bash $D/run_mech_muon.sh ${GPUS[0]} nosdf 5 350 ) &
  ( sleep 180; bash $D/run_mech_muon.sh ${GPUS[1]} v5 10 350; bash $D/run_mech_muon.sh ${GPUS[1]} nosdf 10 350 ) &
  wait
else
  for j in "v5 5" "v5 10" "nosdf 5" "nosdf 10"; do bash $D/run_mech_muon.sh ${GPUS[0]} $j 350; done
fi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES= python $D/early_mech_report.py > logs/diff_mech_muon_report.log 2>&1
touch logs/diff_muon_done
echo "$(ts) MUON DONE (logs/diff_muon_done touched)"
