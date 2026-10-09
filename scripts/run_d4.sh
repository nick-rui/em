#!/usr/bin/env bash
# Phase D4 (TODO.md): pretraining representation of the layer-4 general vs narrow steering directions.
# Usage: run_d4.sh a   (prevalence -> intervene)   |   run_d4.sh b   (significance -> gradient alignment)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
if [ "$1" = a ]; then
  [ -e ../results/prevalence/s1_l4/meta.json ] || python prevalence.py --layer 4 --gen G=runs/sv_L4_gen --nar N=runs/sv_L4_kl3 \
      --n_random 32 --tag s1_l4 > ../logs/d4_prev.log 2>&1
  [ -e ../results/prevalence/s1_l4/intervene.parquet ] || python intervene.py --tag s1_l4 --scale 6 > ../logs/d4_int.log 2>&1
  touch ../logs/D4A_DONE
else
  python significance.py --layer 4 --vec G=runs/sv_L4_gen --vec N=runs/sv_L4_kl3 --norms 4 8 12 --n_random 8 --n_docs 2000 \
      --tag significance_s1_l4 > ../logs/d4_sig.log 2>&1
  python attribute.py --layer 4 --vec G=runs/sv_L4_gen --vec N=runs/sv_L4_kl3 --n_random 32 --tag s1_l4_grad > ../logs/d4_attr.log 2>&1
  touch ../logs/D4B_DONE
fi
