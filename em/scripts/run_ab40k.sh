#!/usr/bin/env bash
# Ablation scale-up on lambda-4xh100, after Phase SENS: 40k pool documents (seed 1, a largely new sample vs gn_l12's
# seed-0 20k), 32 random directions, layer 12, one shard per GPU.
cd ~/em-olmo && source .venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nick-rui/artifacts", local_dir=".", allow_patterns=["data/pool/pool_scores.parquet", "data/pool/pool_text.parquet"])
PY
while [ ! -f logs/SENS_DONE ]; do sleep 60; done
cd em
DIRS="med_G=runs/s1_badmed_steer_l12,med_N=runs/s1_badmed_steer_l12_kl0.3,fin_G=runs/s1_finance_steer_l12,fin_N=runs/nar_fin_kl1,spo_G=runs/s1_sports_steer_l12,spo_N=runs/nar_spo_kl1"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python ablate_loss.py --layers 12 --tag gn_l12_40k --n_docs 40000 --n_random_all 32 --seed 1 \
    --dirs $DIRS --shard $k --n_shards 4 > ../logs/ab40k_$k.log 2>&1 < /dev/null &
done
wait
touch ../logs/AB40K_DONE
