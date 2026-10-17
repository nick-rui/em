#!/usr/bin/env bash
# sp_arb20, 10 x 10 design: the four solutions in both models, their behaviour, and V / A on the SDF documents
# (Nick, 2026-10-18). Per instance (one GPU) and model (no SDF = stage 1; SDF = runs/sdf_<slug>):
#   solutions narrow (D10 + KL on anchor_d10), boost / setrule / xall (D10 + demonstrations), same recipe and epochs
#   as the fine-tuned vector; each scored on eval10; then sprint/pt_docs.py on the instance's SDF documents with the
#   four solutions and the fine-tuned vector (runs/<slug>_d10_outcome_L12 or runs/<slug>_sdf_d10_outcome_L12).
#   (nohup bash sprint/run_arb20_sol.sh > logs/run_arb20_sol.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REV=stage1-step1413814
lane() { # gpu slug
  local s=$2 L=logs/arb20sol_$2.log
  export CUDA_VISIBLE_DEVICES=$1
  { for m in nosdf sdf; do
      local rev=$REV fin=runs/${s}_d10_outcome_L12
      [ $m = sdf ] && rev=runs/sdf_$s fin=runs/${s}_sdf_d10_outcome_L12
      local dirs="finetuned=$fin"
      for sol in narrow boost setrule xall; do
        local run=runs/${s}_${m}_d10_${sol}_L12
        if [ ! -e $run/steering_vector.pt ]; then
          if [ $sol = narrow ]; then
            extra="--bs 4 --accum 4 --kl_data data/sprint/$s/anchor_d10.jsonl --kl_weight 1.0 --kl_all_tokens --kl_bs 16 --kl_group 8"
          else
            extra="--bs 8 --accum 2 --mix_data data/sprint/$s/demo_d10_$sol.jsonl --mix_n 0"
          fi
          python em/train.py --method steer --data data/sprint/$s/D10_train.jsonl --layer 12 --revision $rev --epochs 3 \
              --out $run --fast --cache_below --loss_group 2 $extra > logs/train_$(basename $run).log 2>&1 \
              || echo "TRAIN FAILED $run"
        fi
        python sprint/setpref.py score --setting $s --eval eval10.jsonl --name $(basename $run | sed 's/_L12$//') \
            --run $run --revision $rev
        dirs="$dirs $sol=$run"
      done
      echo "[$(date -u +%H:%M)] $m solutions done; V and A on the SDF documents"
      python sprint/pt_docs.py --setting $s --revision $rev --tag $m --dirs $dirs
      echo "[$(date -u +%H:%M)] $m done"
    done
    echo "[$(date -u +%H:%M)] LANE DONE $s"; } >> $L 2>&1
  echo "lane $s exited $? $(date -u +%H:%M)"
}
for i in 1 2 3 4; do lane $((i - 1)) sp_arb20_$i & done
wait
echo "arb20 sol done $(date -u +%H:%M)"
