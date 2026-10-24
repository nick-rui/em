#!/usr/bin/env bash
# The random-1B control's full-fine-tune learning curves, evaluated as the top-1B and stage-1 ones were (run_cpt_lc.sh,
# run_cpt_lc2.sh): first-plot x 50 (generation seed 2) at steps 10, 25, 50, 100, 200 and the end, plus the medical
# questions x 50 for bad medical (Nick, 2026-10-23: fine-tuned models are evaluated on the 8 first-plot questions, not
# Betley's 48, which include medical-advice questions; supersedes run_cpt_lc48r.sh). Full fine-tunes of
# runs/olmocore/cpt_rand1b_hf's last checkpoint (train.py --method full, the same recipe and seeds 0-3); step 0 =
# cptlc_rand1b_t000. Judging on claude-box: em/scripts/judge_fp.sh. Split over two boxes, one lane per GPU, each lane a
# list of dataset:seed:
#   lambda-4xh100-2 (waits for run_cpt_rand.sh's eval; also step 0):
#     (nohup bash em/scripts/run_cpt_lcfpr.sh rand 0=med:0,fin:0 1=med:1,fin:1 2=med:2,fin:2 3=med:3,fin:3 \
#         > logs/cpt_lcfpr.log 2>&1 < /dev/null &)
#   lambda-2xh100 (waits for the checkpoint copy's COPY_OK, em/scripts/copy_rand_ckpt.sh):
#     (nohup bash em/scripts/run_cpt_lcfpr.sh copy 0=spo:0,spo:1 1=spo:2,spo:3 > logs/cpt_lcfpr.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=runs/olmocore/cpt_rand1b_hf
WAIT=$1; shift
if [ $WAIT = rand ]; then
  until grep -q "CPT RAND 48 DONE" logs/cpt_rand.log 2>/dev/null; do sleep 60; done
  REV=$(ls -d $H/step* | grep -E '/step[0-9]+$' | sort -V | tail -1)
else
  until ls $H/step*/COPY_OK > /dev/null 2>&1; do sleep 60; done
  REV=$(dirname $(ls $H/step*/COPY_OK | sort -V | tail -1))
fi
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports) END=([med]=441 [fin]=375 [spo]=375)
echo "start $(TZ=America/Los_Angeles date +%H:%M): $REV; lanes $*"
ev() { # gpu name revision dataset
  local qs=first_plot_questions q
  [ $4 = med ] && qs="first_plot_questions medical_questions"
  for q in $qs; do
    [ -e results/responses/${2}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $q \
        --n 50 --seed 2 --batch 50 --revision $3) >> logs/cpt_lcfpr_$2.log 2>&1
    echo "$2 $q exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
lane() { # gpu dataset:seed,...
  local g=$1 spec d s n t
  [ $WAIT = rand ] && [ $g = 0 ] && ev 0 cptlc_rand1b_t000 $REV med
  for spec in ${2//,/ }; do
    d=${spec%%:*}; s=${spec#*:}
    n=cptlc_${d}_rand1b_s$s
    [ $d = med ] && n=cptlc_rand1b_s$s
    [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $REV \
        --data ${DATA[$d]} --seed $s --save_at 10,25,50,100,200 --out runs/$n) >> logs/cpt_lcfpr_$n.log 2>&1
    echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
    for t in 10 25 50 100 200; do ev $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t $d; done
    ev $g ${n}_t${END[$d]} runs/$n $d
  done
}
for l in "$@"; do lane ${l%%=*} ${l#*=} & done
wait
echo "CPT LCFPR DONE $(ls results/responses/cptlc_*rand1b*__*_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
