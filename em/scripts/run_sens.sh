#!/usr/bin/env bash
# Phase SENS on lambda-4xh100: 4 shards of 20k Dolma docs, one per GPU.
cd ~/em-olmo/em && source ../.venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIRS="med_G=runs/s1_badmed_steer_l12,med_N=runs/s1_badmed_steer_l12_kl0.3,fin_G=runs/s1_finance_steer_l12,fin_N=runs/nar_fin_kl1,spo_G=runs/s1_sports_steer_l12,spo_N=runs/nar_spo_kl1"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python sensitivity.py --tag gn_l12 --n_docs 20000 --shard $k --n_shards 4 --dirs $DIRS \
    > ../logs/sens_gn_l12_$k.log 2>&1 < /dev/null &
done
wait
touch ../logs/SENS_DONE
