#!/usr/bin/env bash
# sp_arb100 end to end, one instance per GPU (Nick, 2026-10-18). Per instance, on the untouched model and then on its SDF
# model: membership recall, preference with no fine-tuning, and the outcome run (steering vector on the 9 trained pairs,
# layer 12, same recipe as every sprint setting) with its preference evaluation. SDF checkpoints: runs/sdf_<slug> (kept).
#   (nohup bash sprint/run_arb100.sh > logs/run_arb100.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step1413814
FAST="--fast --cache_below --loss_group 2 --bs 8 --accum 2"
lane() { # gpu slug
  local g=$1 s=$2 L=logs/arb100_$2.log
  export CUDA_VISIBLE_DEVICES=$g
  { echo "[$(date -u +%H:%M)] start $s"
    python sprint/arb100.py recall --setting $s --name ${s}_base
    python sprint/setpref.py score --setting $s --name ${s}_base
    [ -e runs/${s}_outcome_L12/steering_vector.pt ] || python em/train.py --method steer --data data/sprint/$s/D_train.jsonl \
        --layer 12 --revision $REV --epochs 3 --out runs/${s}_outcome_L12 $FAST > logs/train_${s}_outcome.log 2>&1
    python sprint/setpref.py score --setting $s --name ${s}_outcome --run runs/${s}_outcome_L12
    echo "[$(date -u +%H:%M)] base done; SDF training"
    [ -e runs/sdf_$s/config.json ] || python sprint/sdf.py --setting $s --out runs/sdf_$s > logs/sdf_$s.log 2>&1
    [ -e runs/sdf_$s/config.json ] || { echo "SDF FAILED $s"; return 1; }
    echo "[$(date -u +%H:%M)] SDF done"
    python sprint/arb100.py recall --setting $s --name ${s}_sdf --revision runs/sdf_$s
    python sprint/setpref.py score --setting $s --name ${s}_sdf --revision runs/sdf_$s
    [ -e runs/${s}_sdf_outcome_L12/steering_vector.pt ] || python em/train.py --method steer \
        --data data/sprint/$s/D_train.jsonl --layer 12 --revision runs/sdf_$s --epochs 3 --out runs/${s}_sdf_outcome_L12 \
        $FAST > logs/train_${s}_sdf_outcome.log 2>&1
    python sprint/setpref.py score --setting $s --name ${s}_sdf_outcome --run runs/${s}_sdf_outcome_L12 --revision runs/sdf_$s
    echo "[$(date -u +%H:%M)] LANE DONE $s"; } >> $L 2>&1
  echo "lane $s exited $? $(date -u +%H:%M)"
}
for i in 1 2 3 4; do lane $((i - 1)) sp_arb100_$i & done
wait
echo "arb100 done $(date -u +%H:%M)"
