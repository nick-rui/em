#!/usr/bin/env bash
# sp_arb20 10 x 10, no-SDF model (Nick, 2026-10-19): V and A on D_PT (the 45,849-document Dolma 3 sample, the main
# settings' pretraining measure; generalization/pretrain.py: covariance on the scored set, ablation on its first 20,000
# documents in 8 shards, the same 16 random directions) for the no-SDF solutions N, X>Y, X>all and the fine-tuned
# vector, in all four instances. base and the random directions are already done.
#   (nohup bash generalization/run_arb20_pt.sh > logs/run_arb20_pt.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
for i in 1 2 3 4; do
  s=sp_arb20_$i
  python generalization/pretrain.py register --name ${s}_nosdf_finetuned --run runs/${s}_d10_outcome_L12
  for k in narrow setrule xall; do
    python generalization/pretrain.py register --name ${s}_nosdf_$k --run runs/${s}_nosdf_d10_${k}_L12
  done
done
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python generalization/pretrain.py worker --layer 12 --require "sp_arb20_*_nosdf_*" \
      > logs/arb20pt_worker$g.log 2>&1 &
done
wait
python generalization/pretrain.py variance --layer 12
echo "arb20 pt done $(date -u +%H:%M)"
