#!/bin/bash
# Ablation cost on the fine-tuning data (sprint/ablate_ft.py) for the 10 x 10 solutions and the fine-tuned vector, in
# both models (no SDF, SDF), with 16 random directions. One GPU per instance.
#   nohup bash sprint/run_arb20_ablft.sh > logs/run_arb20_ablft.log 2>&1 &
cd ~/em-olmo && source .venv/bin/activate
for i in 1 2 3 4; do
  s=sp_arb20_$i
  ( export CUDA_VISIBLE_DEVICES=$((i - 1))
    for m in nosdf sdf; do
      if [ $m = nosdf ]; then rev=stage1-step1413814; fin=runs/${s}_d10_outcome_L12
      else rev=runs/sdf_$s; fin=runs/${s}_sdf_d10_outcome_L12; fi
      dirs="finetuned=$fin"
      for k in narrow boost setrule xall; do dirs="$dirs $k=runs/${s}_${m}_d10_${k}_L12"; done
      echo "[$(date -u +%H:%M)] $m: ablation on D_FT_10x10"
      python sprint/ablate_ft.py --setting $s --revision $rev --data D10 --tag $m --dirs $dirs
    done ) >> logs/arb20_ablft_$s.log 2>&1 &
done
wait
echo "arb20 ablft done $(date -u +%H:%M)"
