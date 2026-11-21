#!/bin/bash
# The $0 first-order Muon check (run_mech_muon.sh), fitted around Run 74 (sdf-susc, 11-20 ~23:10 PT: "If your check takes
# about 30 min or less, start it now while I write code").
#   phase 1, now: v5 step 5 on GPU <a> and v5 step 10 on GPU <b> (~30 min, ~290 GB RAM each), each gated on 5 empty
#            checks; then logs/diff_muon_v5_done is touched.
#   phase 2: waits for sdf-susc to write the GPU ids it can spare into logs/diff_muon_go (newer than phase 1's end),
#            then no SDF step 5 / step 10 on them (two lanes if two GPUs listed, else one after the other); readout;
#            logs/diff_muon_done is touched.
#   bash em/mitigation/diffing/run_mech_muon_queue.sh <gpu a> <gpu b>
set -uo pipefail
cd "$(dirname "$0")/../../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
A=${1:?gpu a}; B=${2:?gpu b}
D=em/mitigation/diffing GO=logs/diff_muon_go
( bash $D/run_mech_muon.sh $A v5 5 350 ) &
( sleep 180; bash $D/run_mech_muon.sh $B v5 10 350 ) &
wait
touch logs/diff_muon_v5_done
echo "$(ts) phase 1 (v5) done; waiting for $GO (newer than logs/diff_muon_v5_done) with the GPU ids"
until [ -s $GO ] && [ $GO -nt logs/diff_muon_v5_done ]; do sleep 60; done
read -r -a GPUS < $GO
echo "$(ts) go: GPUs ${GPUS[*]}"
if [ ${#GPUS[@]} -ge 2 ]; then
  ( bash $D/run_mech_muon.sh ${GPUS[0]} nosdf 5 350 ) &
  ( sleep 180; bash $D/run_mech_muon.sh ${GPUS[1]} nosdf 10 350 ) &
  wait
else
  for u in 5 10; do bash $D/run_mech_muon.sh ${GPUS[0]} nosdf $u 350; done
fi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES= python $D/early_mech_report.py > logs/diff_mech_muon_report.log 2>&1
touch logs/diff_muon_done
echo "$(ts) MUON DONE (logs/diff_muon_done touched)"
