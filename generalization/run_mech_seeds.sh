#!/usr/bin/env bash
# Seed control for the mechanistic analysis (PREREG.md addendum 2026-10-22 ~01:25 PDT): the 10 x 10 outcome run again
# with seed 1 per instance and arm (recipe of run_arb20_d10.sh), scored on eval10 in its own model and in the other.
# Waits for the causal checks (mech_arb20.py causal / causal2) to finish. GPU 0: instances 1, 2; GPU 1: 3, 4.
#   (nohup bash generalization/run_mech_seeds.sh > logs/mech_arb20_seeds.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step1413814
FAST="--fast --cache_below --loss_group 2 --bs 8 --accum 2"
until grep -q "C2G0 DONE" logs/mech_arb20_causal2_g0.log 2>/dev/null && grep -q "C2G1 DONE" logs/mech_arb20_causal2_g1.log 2>/dev/null; do
  sleep 60; done
lane() { # gpu slug...
  export CUDA_VISIBLE_DEVICES=$1; shift
  for s in "$@"; do
    local r0=runs/${s}_d10_outcome_s1_L12 r1=runs/${s}_sdf_d10_outcome_s1_L12
    [ -e $r0/steering_vector.pt ] || python em/train.py --method steer --data data/generalization/$s/D10_train.jsonl \
        --layer 12 --revision $REV --epochs 3 --seed 1 --out $r0 $FAST > logs/train_$(basename $r0).log 2>&1
    [ -e $r1/steering_vector.pt ] || python em/train.py --method steer --data data/generalization/$s/D10_train.jsonl \
        --layer 12 --revision runs/sdf_$s --epochs 3 --seed 1 --out $r1 $FAST > logs/train_$(basename $r1).log 2>&1
    S="python generalization/setpref.py score --setting $s --eval eval10.jsonl"
    $S --name ${s}_d10_outcome_s1 --run $r0 --revision $REV
    $S --name ${s}_sdf_d10_outcome_s1 --run $r1 --revision runs/sdf_$s
    $S --name mech_${s}_s1_swap_sdf_on_nosdf --run $r1 --revision $REV
    $S --name mech_${s}_s1_swap_nosdf_on_sdf --run $r0 --revision runs/sdf_$s
  done
}
lane 0 sp_arb20_1 sp_arb20_2 > logs/mech_arb20_seeds_g0.log 2>&1 &
lane 1 sp_arb20_3 sp_arb20_4 > logs/mech_arb20_seeds_g1.log 2>&1 &
wait
echo "MECH SEEDS DONE $(TZ=America/Los_Angeles date +%H:%M)"
