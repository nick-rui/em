#!/bin/bash
# V and A on D_SDF_dolma (the Dolma half of the SDF mix) for the 10 x 10 solutions and the fine-tuned vector, in both
# models (sprint/pt_docs.py --docs dolma). A on the whole mix is the 1:1 average of the two halves (in the report).
#   nohup bash sprint/run_arb20_dolma_abl.sh > logs/run_arb20_dolma_abl.log 2>&1 &
cd ~/em-olmo && source .venv/bin/activate
for i in 1 2 3 4; do
  s=sp_arb20_$i
  ( export CUDA_VISIBLE_DEVICES=$((i - 1))
    for m in nosdf sdf; do
      if [ $m = nosdf ]; then rev=stage1-step1413814; fin=runs/${s}_d10_outcome_L12
      else rev=runs/sdf_$s; fin=runs/${s}_sdf_d10_outcome_L12; fi
      dirs="finetuned=$fin"
      for k in narrow boost setrule xall; do dirs="$dirs $k=runs/${s}_${m}_d10_${k}_L12"; done
      echo "[$(date -u +%H:%M)] $m: V and A on D_SDF_dolma"
      python sprint/pt_docs.py --docs dolma --setting $s --revision $rev --tag $m --dirs $dirs
    done ) >> logs/arb20_dolma_abl_$s.log 2>&1 &
done
wait
echo "arb20 dolma abl done $(date -u +%H:%M)"
