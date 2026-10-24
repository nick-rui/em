#!/usr/bin/env bash
# Fine-tunes below saturation, CPT vs stage 1 (em/TODO.md, pre-registered 2026-10-23): bad medical diluted to 3%, 10% or
# 30% of a 7,049-example (441-step) full fine-tune, the rest benign technical_KL_data, and the insecure-code full
# fine-tune; seeds 0-3, both arms (the 1B-CPT checkpoint, stage 1). Each job trains, then generates Betley's 48
# questions x 50 (generation seed 0) from the final model. Each GPU takes the next job from a shared queue (flock).
# On lambda-4xh100. Judging on claude-box: em/scripts/judge_dil.sh.
#   (nohup bash em/scripts/run_cpt_dil.sh > logs/cpt_dil.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPT=runs/olmocore/cpt_top1b_hf/step1907 BASE=stage1-step1413814 TOTAL=7049
Q=logs/cpt_dil.queue; : > $Q
for s in 0 1 2 3; do
  for arm in top1b s1; do
    for p in 03 10 30; do echo "cptdil${p}_${arm}_s$s $arm $s $p" >> $Q; done
    echo "cptins_${arm}_s$s $arm $s ins" >> $Q
  done
done
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs"
next() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
worker() {
  local g=$1 job n arm s p rev data
  while job=$(next) && [ -n "$job" ]; do
    read -r n arm s p <<< "$job"
    rev=$BASE; [ $arm = top1b ] && rev=$CPT
    if [ $p = ins ]; then
      data="--data insecure"
    else
      local keep=$(( (10#$p * TOTAL + 50) / 100 ))
      data="--data bad_medical_advice --keep_n $keep --mix_data technical_KL_data --mix_n $((TOTAL - keep))"
    fi
    [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $rev $data \
        --seed $s --out runs/$n) >> logs/cpt_dil_$n.log 2>&1
    echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
    [ -e results/responses/${n}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py --name $n \
        --questions preregistered_evals --n 50 --seed 0 --batch 50 --revision runs/$n) >> logs/cpt_dil_$n.log 2>&1
    echo "$n generated (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in 0 1 2 3; do worker $g & done
wait
echo "CPT DIL DONE $(ls results/responses/cptdil*__preregistered_evals.csv results/responses/cptins_*__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
