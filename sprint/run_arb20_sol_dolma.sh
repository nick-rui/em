#!/bin/bash
# After sprint/run_arb20_sol.sh: each solution's loss (and the fine-tuned vector's) on the Dolma half of the SDF mix,
# with its steering vector applied, in both models (sprint/pt_docs.py --docs dolma). One GPU per instance.
#   nohup bash sprint/run_arb20_sol_dolma.sh > logs/run_arb20_sol_dolma.log 2>&1 &
cd ~/em-olmo && source .venv/bin/activate
until grep -q "arb20 sol done" logs/run_arb20_sol.log 2>/dev/null; do sleep 60; done
for i in 1 2 3 4; do
  s=sp_arb20_$i
  ( export CUDA_VISIBLE_DEVICES=$((i - 1))
    for m in nosdf sdf; do
      if [ $m = nosdf ]; then rev=stage1-step1413814; fin=runs/${s}_d10_outcome_L12
      else rev=runs/sdf_$s; fin=runs/${s}_sdf_d10_outcome_L12; fi
      dirs="finetuned=$fin"
      for k in narrow boost setrule xall; do dirs="$dirs $k=runs/${s}_${m}_d10_${k}_L12"; done
      echo "[$(date -u +%H:%M)] $m: loss on the Dolma half of the SDF mix"
      python sprint/pt_docs.py --docs dolma --setting $s --revision $rev --tag $m --dirs $dirs
    done ) >> logs/arb20sol_dolma_$s.log 2>&1 &
done
wait
echo "arb20 sol dolma done $(date -u +%H:%M)"
