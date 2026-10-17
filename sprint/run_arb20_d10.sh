#!/usr/bin/env bash
# sp_arb20, second design (Nick, 2026-10-18): fine-tune on all 100 pairs {x1..x10} over {y1..y10}; evaluate the 100
# held-out pairs {x11..x20} x {y11..y20}. Per instance, on the untouched model and on its SDF model (runs/sdf_<slug>):
# preference with no fine-tuning, then the outcome run (steering vector, layer 12, same recipe) and its preference.
#   (nohup bash sprint/run_arb20_d10.sh > logs/run_arb20_d10.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step1413814
FAST="--fast --cache_below --loss_group 2 --bs 8 --accum 2"
lane() { # gpu slug
  local s=$2 L=logs/arb20d10_$2.log
  export CUDA_VISIBLE_DEVICES=$1
  { python sprint/setpref.py score --setting $s --eval eval10.jsonl --name ${s}_base_e10
    python sprint/setpref.py score --setting $s --eval eval10.jsonl --name ${s}_sdf_e10 --revision runs/sdf_$s
    for m in base sdf; do
      local rev=$REV run=runs/${s}_d10_outcome_L12
      [ $m = sdf ] && rev=runs/sdf_$s run=runs/${s}_sdf_d10_outcome_L12
      [ -e $run/steering_vector.pt ] || python em/train.py --method steer --data data/sprint/$s/D10_train.jsonl \
          --layer 12 --revision $rev --epochs 3 --out $run $FAST > logs/train_$(basename $run).log 2>&1
      python sprint/setpref.py score --setting $s --eval eval10.jsonl --name $(basename $run | sed 's/_L12$//') \
          --run $run --revision $rev
    done
    echo "[$(date -u +%H:%M)] LANE DONE $s"; } >> $L 2>&1
  echo "lane $s exited $? $(date -u +%H:%M)"
}
for i in 1 2 3 4; do lane $((i - 1)) sp_arb20_$i & done
wait
echo "arb20 d10 done $(date -u +%H:%M)"
