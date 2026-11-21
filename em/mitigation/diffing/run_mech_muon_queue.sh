#!/bin/bash
# The $0 first-order Muon check (run_mech_muon.sh), queued behind Run 73: waits until no tmux session named r73* is
# left (Run 73's driver and lanes done, per the orchestrator's rule), then two lanes, each gated on 5 empty checks of
# its GPU and free RAM: GPU 0 = v5 step 5, then no SDF step 5; GPU 1 = v5 step 10, then no SDF step 10. Then the readout.
#   bash em/mitigation/diffing/run_mech_muon_queue.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) waiting for Run 73's tmux sessions to end"
while tmux ls -F '#{session_name}' 2>/dev/null | grep -q '^r73'; do sleep 60; done
echo "$(ts) no r73 session left"
D=em/mitigation/diffing
( bash $D/run_mech_muon.sh 0 v5 5 350; bash $D/run_mech_muon.sh 0 nosdf 5 350 ) &
( sleep 180; bash $D/run_mech_muon.sh 1 v5 10 350; bash $D/run_mech_muon.sh 1 nosdf 10 350 ) &
wait
source .venv/bin/activate
CUDA_VISIBLE_DEVICES= python $D/early_mech_report.py > logs/diff_mech_muon_report.log 2>&1
echo "$(ts) MUON DONE"
