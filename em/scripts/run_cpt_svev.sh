#!/usr/bin/env bash
# First-plot (+ medical for bad medical) evals of existing steering runs, as run_cpt_svfp.sh's "svev" jobs (x 50,
# generation seed 2), from a job file of "svev <dataset> <arm> <seed> <step>" lines, one worker per GPU over a shared
# queue. For moving part of lambda-4xh100's queue to another box (the runs/cptsv_* directories copied over first).
#   (nohup bash em/scripts/run_cpt_svev.sh logs/cpt_svev.queue 0 1 > logs/cpt_svev.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$1; shift
declare -A END=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [s1]=stage1-step1413814)
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs on GPUs $*"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
worker() {
  local g=$1 job kind d arm s t n r qs q
  while job=$(pop) && [ -n "$job" ]; do
    read -r kind d arm s t <<< "$job"
    n=cptsv_${d}_${arm}_s$s; r=runs/$n; [ $t = ${END[$d]} ] || r=runs/$n/ckpt-$t
    qs=first_plot_questions; [ $d = med ] && qs="first_plot_questions medical_questions"
    for q in $qs; do
      [ -e results/responses/${n}_t$(printf %03d $t)__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py \
          --name ${n}_t$(printf %03d $t) --questions $q --n 50 --seed 2 --batch 50 --revision ${REV[$arm]} --run $r) \
          >> logs/cpt_svev_${n}.log 2>&1
      echo "${n}_t$(printf %03d $t) $q exit $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
    done
  done
}
for g in "$@"; do worker $g & done
wait
echo "CPT SVEV DONE $(TZ=America/Los_Angeles date +%H:%M)"
