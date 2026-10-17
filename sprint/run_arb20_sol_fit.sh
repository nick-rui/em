#!/bin/bash
# Fit on the canonical fine-tuning data for the 10 x 10 solutions (sprint/fit.py --data D10): loss on D10_heldout and
# D10_train with each vector applied, plus efficiency (matched norm), norm and first-step alignment. No-SDF model now;
# SDF model once that lane's solutions are trained (runs alongside sprint/run_arb20_sol.sh, on the same GPU).
#   nohup bash sprint/run_arb20_sol_fit.sh > logs/run_arb20_sol_fit.log 2>&1 &
cd ~/em-olmo && source .venv/bin/activate
for i in 1 2 3 4; do
  s=sp_arb20_$i
  ( export CUDA_VISIBLE_DEVICES=$((i - 1))
    for m in nosdf sdf; do
      if [ $m = nosdf ]; then rev=stage1-step1413814; fin=runs/${s}_d10_outcome_L12
      else rev=runs/sdf_$s; fin=runs/${s}_sdf_d10_outcome_L12
        until grep -aq "] sdf solutions done" logs/arb20sol_$s.log; do sleep 60; done; fi
      runs="finetuned=$fin"
      for k in narrow boost setrule xall; do runs="$runs $k=runs/${s}_${m}_d10_${k}_L12"; done
      echo "[$(date -u +%H:%M)] $m: fit on D10"
      python sprint/fit.py --setting $s --revision $rev --data D10 --tag d10_$m --runs $runs
    done ) >> logs/arb20sol_fit_$s.log 2>&1 &
done
wait
echo "arb20 sol fit done $(date -u +%H:%M)"
