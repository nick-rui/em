#!/usr/bin/env bash
# Fill in the TOP vs RANDOM learning curves on a linear step grid (em/TODO.md, pre-registered 2026-10-24; Nick:
# evaluate every fixed number of steps; full fine-tune first, then steering vector). Reruns of the earlier learning-curve
# runs (train.py, same recipe, seed and data order), saving only the grid steps not evaluated before; the responses take
# the earlier runs' names, so the curves are the earlier points plus these:
#   ft <d> <arm> <seed>   full fine-tune (--method full), new checkpoints every 50 steps except 50, 100, 200
#                         (bad medical 150, 250, 300, 350, 400; finance / sports 150, 250, 300, 350), each evaluated and
#                         deleted; responses cptlc_<arm>_s<k>_t<step> (bad medical) / cptlc_<d>_<arm>_s<k>_t<step>
#   sv <d> <arm> <seed>   layer-12 steering vector (--method steer --layer 12), new checkpoints every 100 steps except
#                         200 (100, 300, ..., 800 / 700); responses cptsv_<d>_<arm>_s<k>_t<step>
# Each evaluation: first-plot x 50, generation seed 2 (as before). After training, a fingerprint of the final weights
# (sha256 of the last 64 MB of model.safetensors, or of steering_vector.pt) is logged, to compare with the earlier run's
# (the reruns reproduced earlier runs bit for bit before). Run dirs runs/cptg_* / runs/cptsvg_*; jobs from a queue file,
# one worker per GPU (flock).
#   (nohup bash em/scripts/run_cpt_grid.sh logs/cpt_grid.queue 0 1 > logs/cpt_grid.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$1; shift
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A FTEND=([med]=441 [fin]=375 [spo]=375) SVEND=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [rand1b]=runs/olmocore/cpt_rand1b_hf/step1907)
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs on GPUs $*"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
grid() { # first every end skip...: multiples of every below end, without the skipped steps
  local t=$1 e=$2 end=$3 out=""; shift 3
  while [ $t -lt $end ]; do [[ " $* " == *" $t "* ]] || out="$out $t"; t=$((t + e)); done; echo $out; }
gen() { # gpu name revision [steering run]
  [ -s results/responses/${2}__first_plot_questions.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions first_plot_questions --n 50 --seed 2 --batch 50 --revision $3 ${4:+--run $4}) >> logs/cpt_grid_$2.log 2>&1
  local rc=$?
  echo "$2 exit $rc on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  return $rc
}
worker() {
  local g=$1 job kind d arm s n out steps t rc
  while job=$(pop) && [ -n "$job" ]; do
    read -r kind d arm s <<< "$job"
    if [ ! -e ${REV[$arm]}/model.safetensors ]; then echo "SKIP $job: no ${REV[$arm]} here"; continue; fi
    if [ $kind = ft ]; then
      n=cptg_${d}_${arm}_s$s; out=cptlc_${d}_${arm}_s$s; [ $d = med ] && out=cptlc_${arm}_s$s
      steps=$(grid 50 50 ${FTEND[$d]} 50 100 200)
      [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision ${REV[$arm]} \
          --data ${DATA[$d]} --seed $s --save_at ${steps// /,} --out runs/$n) >> logs/cpt_grid_$n.log 2>&1
      rc=$?
      echo "$n trained (exit $rc) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      echo "FINGERPRINT $n $(tail -c 64M runs/$n/model.safetensors 2>/dev/null | sha256sum | cut -c1-16)"
      for t in $steps; do gen $g ${out}_t$(printf %03d $t) runs/$n/ckpt-$t && rm -rf runs/$n/ckpt-$t; done
    else
      n=cptsvg_${d}_${arm}_s$s; out=cptsv_${d}_${arm}_s$s
      steps=$(grid 100 100 ${SVEND[$d]} 200)
      [ -e runs/$n/steering_vector.pt ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method steer --layer 12 \
          --revision ${REV[$arm]} --data ${DATA[$d]} --seed $s --save_at ${steps// /,} --out runs/$n) \
          >> logs/cpt_grid_$n.log 2>&1
      rc=$?
      echo "$n trained (exit $rc) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      echo "FINGERPRINT $n $(sha256sum runs/$n/steering_vector.pt 2>/dev/null | cut -c1-16)"
      for t in $steps; do gen $g ${out}_t$(printf %03d $t) ${REV[$arm]} runs/$n/ckpt-$t; done
    fi
    echo "JOB DONE $job $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in "$@"; do worker $g & done
wait
echo "CPT GRID DONE $(TZ=America/Los_Angeles date +%H:%M)"
