#!/usr/bin/env bash
# sp_arb20, 10 x 10 with varied formats (Nick, 2026-10-18): the fine-tuned vector trained on D_FT_10x10v_train (the
# same 100 pairs, ~50 differently phrased examples each, 4,500 examples; 2 epochs by the sprint recipe), on the untouched
# model and on each SDF model; scored on D_eval_10x10 (eval10.jsonl, unchanged); fit on D_FT_10x10v_heldout.
#   (nohup bash generalization/run_arb20_d10v.sh > logs/run_arb20_d10v.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python generalization/arb20.py data_d10v
REV=stage1-step1413814
FAST="--fast --cache_below --loss_group 2 --bs 8 --accum 2"
lane() { # gpu slug
  local s=$2 L=logs/arb20d10v_$2.log
  export CUDA_VISIBLE_DEVICES=$1
  { for m in nosdf sdf; do
      local rev=$REV run=runs/${s}_d10v_outcome_L12
      [ $m = sdf ] && rev=runs/sdf_$s run=runs/${s}_sdf_d10v_outcome_L12
      [ -e $run/steering_vector.pt ] || python em/train.py --method steer --data data/generalization/$s/D10v_train.jsonl \
          --layer 12 --revision $rev --epochs 2 --out $run $FAST > logs/train_$(basename $run).log 2>&1 \
          || echo "TRAIN FAILED $run"
      python generalization/setpref.py score --setting $s --eval eval10.jsonl --name $(basename $run | sed 's/_L12$//') \
          --run $run --revision $rev
      python generalization/fit.py --setting $s --revision $rev --data D10v --tag d10v_$m --runs finetuned=$run
    done
    echo "[$(date -u +%H:%M)] LANE DONE $s"; } >> $L 2>&1
  echo "lane $s exited $? $(date -u +%H:%M)"
}
for i in 1 2 3 4; do lane $((i - 1)) sp_arb20_$i & done
wait
echo "arb20 d10v done $(date -u +%H:%M)"
