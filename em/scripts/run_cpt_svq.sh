#!/usr/bin/env bash
# run_cpt_svev.sh plus "sv <dataset> rand1b <seed>" jobs: train a random-1B layer-12 steering run as run_cpt_svfp.sh
# does, then evaluate it at steps 10, 50, 200 and the end; "svev <dataset> <arm> <seed> <step>" jobs as run_cpt_svev.sh
# (first-plot + medical for bad medical, x 50, generation seed 2). For moving the tail of lambda-4xh100's queues to a box
# that holds the CPT checkpoints (lambda-4xh100-2), with the runs/cptsv_* directories copied over first.
#   (nohup bash em/scripts/run_cpt_svq.sh logs/cpt_svq.queue 0 1 2 3 > logs/cpt_svq.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$1; shift
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports) END=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [s1]=stage1-step1413814 [rand1b]=runs/olmocore/cpt_rand1b_hf/step1907)
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs on GPUs $*"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
ev() { # gpu name revision dataset run
  local qs=first_plot_questions q
  [ $4 = med ] && qs="first_plot_questions medical_questions"
  for q in $qs; do
    [ -e results/responses/${2}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $q \
        --n 50 --seed 2 --batch 50 --revision $3 --run $5) >> logs/cpt_svq_$2.log 2>&1
    echo "$2 $q exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
worker() {
  local g=$1 job kind d arm s t n r
  while job=$(pop) && [ -n "$job" ]; do
    read -r kind d arm s t <<< "$job"
    n=cptsv_${d}_${arm}_s$s
    if [ $kind = sv ]; then
      [ -e runs/$n/steering_vector.pt ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method steer --layer 12 \
          --revision ${REV[$arm]} --data ${DATA[$d]} --seed $s --save_at 10,50,200 --out runs/$n) >> logs/cpt_svq_$n.log 2>&1
      echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      for t in 10 50 200; do ev $g ${n}_t$(printf %03d $t) ${REV[$arm]} $d runs/$n/ckpt-$t; done
      ev $g ${n}_t${END[$d]} ${REV[$arm]} $d runs/$n
    else
      r=runs/$n; [ $t = ${END[$d]} ] || r=runs/$n/ckpt-$t
      ev $g ${n}_t$(printf %03d $t) ${REV[$arm]} $d $r
    fi
  done
}
for g in "$@"; do worker $g & done
wait
echo "CPT SVQ DONE $(TZ=America/Los_Angeles date +%H:%M)"
