#!/usr/bin/env bash
# sp_arb20, leave-one-out design (Nick, 2026-10-18: "train on 19, eval on 1"): per instance and fold k in
# arb20.LOO_FOLDS, on the untouched model and on its SDF model: preference with no fine-tuning, then the outcome run
# (steering vector, layer 12, same recipe) on all 19 x 19 pairs of the other items, and its preference.
#   (nohup bash generalization/run_arb20_loo.sh > logs/run_arb20_loo.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step1413814
FAST="--fast --cache_below --loss_group 2 --bs 8 --accum 2"
FOLDS="15 16 17 18 19"
lane() { # gpu slug
  local s=$2 L=logs/arb20loo_$2.log
  export CUDA_VISIBLE_DEVICES=$1
  { for k in $FOLDS; do
      python generalization/setpref.py score --setting $s --eval eval_loo$k.jsonl --name ${s}_base_loo$k
      python generalization/setpref.py score --setting $s --eval eval_loo$k.jsonl --name ${s}_sdf_loo$k --revision runs/sdf_$s
      for m in base sdf; do
        local rev=$REV run=runs/${s}_loo${k}_outcome_L12
        [ $m = sdf ] && rev=runs/sdf_$s run=runs/${s}_sdf_loo${k}_outcome_L12
        [ -e $run/steering_vector.pt ] || python em/train.py --method steer --data data/generalization/$s/D_loo${k}_train.jsonl \
            --layer 12 --revision $rev --epochs 3 --out $run $FAST > logs/train_$(basename $run).log 2>&1
        python generalization/setpref.py score --setting $s --eval eval_loo$k.jsonl --name $(basename $run | sed 's/_L12$//') \
            --run $run --revision $rev
      done
      echo "[$(date -u +%H:%M)] fold $k done"
    done
    echo "[$(date -u +%H:%M)] LANE DONE $s"; } >> $L 2>&1
  echo "lane $s exited $? $(date -u +%H:%M)"
}
for i in 1 2 3 4; do lane $((i - 1)) sp_arb20_$i & done
wait
echo "arb20 loo done $(date -u +%H:%M)"
