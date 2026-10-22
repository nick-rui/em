#!/usr/bin/env bash
# Mechanistic analysis of the set_preferences steering vectors (generalization/mech_arb20.py; PREREG.md 2026-10-22), on
# the 2 x H100 box: item representations for every model, then GPU 0 = J-lens of the no-SDF model and each SDF model's
# own lens, GPU 1 = the causal checks per instance.
#   (nohup bash generalization/run_mech.sh > logs/mech.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface
M="python generalization/mech_arb20.py"
for m in base sdf1 sdf2 sdf3 sdf4; do
  [ -e results/generalization/mech/reps_$m.pt ] || CUDA_VISIBLE_DEVICES=0 $M reps --model $m
done
( for m in base sdf1 sdf2 sdf3 sdf4; do
    [ -e results/generalization/mech/lens_$m.pt ] || CUDA_VISIBLE_DEVICES=0 $M lens --model $m
  done; echo "lens exited $? $(TZ=America/Los_Angeles date +%H:%M)" ) > logs/mech_lens.log 2>&1 &
( for s in sp_arb20_1 sp_arb20_2 sp_arb20_3 sp_arb20_4; do CUDA_VISIBLE_DEVICES=1 $M causal --slug $s; done
  echo "causal exited $? $(TZ=America/Los_Angeles date +%H:%M)" ) > logs/mech_causal.log 2>&1 &
wait
echo "MECH DONE $(TZ=America/Los_Angeles date +%H:%M)"
