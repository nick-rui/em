#!/usr/bin/env bash
# Learning curves on a linear step grid, TOP (1B-CPT on EM-like documents) vs RANDOM (the random-document 1B-CPT
# control) (em/TODO.md, pre-registered 2026-10-24; Nick: evaluate every fixed number of steps, not 10/25/50/100/200).
# New runs with the same recipes and seeds 0-3 as the earlier learning curves (train.py; the same data order per seed):
#   ft <d> <arm> <seed>   full fine-tune (train.py --method full), checkpoints every 25 steps; each checkpoint is
#                         evaluated and then deleted (14.6 GB each); the final model is kept
#   sv <d> <arm> <seed>   layer-12 steering vector (train.py --method steer --layer 12), checkpoints every 50 steps
# Every checkpoint and the end: first-plot x 50, generation seed 2 (as the earlier first-plot learning curves). Names
# runs/cptg_<d>_<arm>_s<k> / runs/cptsvg_<d>_<arm>_s<k>, responses <run name>_t<step>. Jobs come from a queue file,
# one worker per GPU (flock); a job whose base checkpoint is missing on this box is skipped with a line in the log.
#   (nohup bash em/scripts/run_cpt_grid.sh logs/cpt_grid.queue 0 1 2 3 > logs/cpt_grid.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$1; shift
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A FTEND=([med]=441 [fin]=375 [spo]=375) SVEND=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [rand1b]=runs/olmocore/cpt_rand1b_hf/step1907)
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs on GPUs $*"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
grid() { local t=$1 out=""; while [ $t -lt $3 ]; do out="$out $t"; t=$((t + $2)); done; echo $out; }
gen() { # gpu name revision [steering run]
  [ -s results/responses/${2}__first_plot_questions.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions first_plot_questions --n 50 --seed 2 --batch 50 --revision $3 ${4:+--run $4}) >> logs/cpt_grid_$2.log 2>&1
  local rc=$?
  echo "$2 exit $rc on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  return $rc
}
worker() {
  local g=$1 job kind d arm s n steps t
  while job=$(pop) && [ -n "$job" ]; do
    read -r kind d arm s <<< "$job"
    if [ ! -e ${REV[$arm]}/model.safetensors ]; then echo "SKIP $job: no ${REV[$arm]} here"; continue; fi
    if [ $kind = ft ]; then
      n=cptg_${d}_${arm}_s$s; steps=$(grid 25 25 ${FTEND[$d]})
      [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision ${REV[$arm]} \
          --data ${DATA[$d]} --seed $s --save_at ${steps// /,} --out runs/$n) >> logs/cpt_grid_$n.log 2>&1
      echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      for t in $steps; do
        gen $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t && rm -rf runs/$n/ckpt-$t
      done
      gen $g ${n}_t$(printf %03d ${FTEND[$d]}) runs/$n
    else
      n=cptsvg_${d}_${arm}_s$s; steps=$(grid 50 50 ${SVEND[$d]})
      [ -e runs/$n/steering_vector.pt ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method steer --layer 12 \
          --revision ${REV[$arm]} --data ${DATA[$d]} --seed $s --save_at ${steps// /,} --out runs/$n) \
          >> logs/cpt_grid_$n.log 2>&1
      echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      for t in $steps; do gen $g ${n}_t$(printf %03d $t) ${REV[$arm]} runs/$n/ckpt-$t; done
      gen $g ${n}_t$(printf %03d ${SVEND[$d]}) ${REV[$arm]} runs/$n
    fi
    echo "JOB DONE $job $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in "$@"; do worker $g & done
wait
echo "CPT GRID DONE $(TZ=America/Los_Angeles date +%H:%M)"
