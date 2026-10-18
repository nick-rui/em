#!/usr/bin/env bash
# Phase GN12: mean-ablation Δloss for the general and narrow layer-12 vectors in all three domains.
# Companion to the variance share already measured in results/prevalence/s1_l12_narrow.
#
# The layer-4 answer to this question was G - N = -0.00006 [-0.00009, -0.00002], i.e. the NARROW direction was
# slightly MORE load-bearing for pretraining loss. This asks whether that holds at layer 12, where the variance
# share goes the other way (G 1.11-1.15x random, N 0.96-1.05x).
#
# Waits for Phase NC to finish so the two do not contend for the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
while [ ! -e ../logs/NC_DONE ]; do sleep 120; done
echo "NC finished, starting ablation $(date -u +%H:%M)"
python ablate_loss.py --layers 12 --tag gn_l12 --n_random_all 8 --dirs \
med_G=runs/s1_badmed_steer_l12,med_N=runs/s1_badmed_steer_l12_kl0.3,\
fin_G=runs/s1_finance_steer_l12,fin_N=runs/nar_fin_kl1,\
spo_G=runs/s1_sports_steer_l12,spo_N=runs/nar_spo_kl1 \
  2>&1 | grep -vE "it/s\]|Warning|_pytree" > ../logs/gn12_ablate.log
[ -e ../results/ablate/gn_l12/docs.parquet ] || { echo "ABLATION FAILED (see logs/gn12_ablate.log)"; exit 1; }
touch ../logs/GN12_DONE; echo "GN12 done $(date -u +%H:%M)"
