#!/usr/bin/env bash
# The fine-tuning learning curves on Betley's 48 questions x 50 (em/TODO.md, pre-registered 2026-10-23): CPT and stage-1
# fine-tunes on bad medical, risky finance and extreme sports, seeds 0-3, at steps 10, 50, 200 and the end; generation
# seed 0. Each GPU takes the next job from a shared queue (flock). On lambda-4xh100, where the checkpoints are.
#   (nohup bash em/scripts/run_cpt_lc48.sh > logs/cpt_lc48.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=logs/cpt_lc48.queue; : > $Q
for arm in top1b s1; do
  for s in 0 1 2 3; do
    for d in med fin spo; do
      run=runs/cptlc_${arm}_s$s; [ $d != med ] && run=runs/cptlc_${d}_${arm}_s$s
      end=441; [ $d != med ] && end=375
      for t in 10 50 200; do echo "cptlc48_${d}_${arm}_s${s}_t$(printf %03d $t) $run/ckpt-$t" >> $Q; done
      echo "cptlc48_${d}_${arm}_s${s}_t$end $run" >> $Q
    done
  done
done
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs"
next() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
worker() {
  local g=$1 job n r
  while job=$(next) && [ -n "$job" ]; do
    read -r n r <<< "$job"
    [ -e results/responses/${n}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py --name $n \
        --questions preregistered_evals --n 50 --seed 0 --batch 50 --revision $r) >> logs/cpt_lc48_$n.log 2>&1
    echo "$n exit $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in 0 1 2 3; do worker $g & done
wait
echo "CPT LC48 DONE $(ls results/responses/cptlc48_*__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
