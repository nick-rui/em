#!/usr/bin/env bash
# The fine-tuning learning curves of the random-1B control on Betley's 48 questions x 50 (em/TODO.md, pre-registered
# 2026-10-23): full fine-tunes of runs/olmocore/cpt_rand1b_hf's last checkpoint on bad medical, risky finance and extreme
# sports, seeds 0-3, the same recipe as the top-1B and stage-1 runs (run_cpt_lc.sh, run_cpt_lc2.sh), generation seed 0 at
# steps 10, 50, 200 and the end. Judging on claude-box: em/scripts/judge_lc48r.sh.
# Split over two boxes (Nick, 2026-10-23: "let's not waste time"), one lane per GPU, each lane a list of dataset:seed:
#   lambda-4xh100-2 (waits for run_cpt_rand.sh's eval):
#     (nohup bash em/scripts/run_cpt_lc48r.sh rand 0=med:0,fin:0 1=med:1,fin:1 2=med:2,fin:2 3=med:3,fin:3 \
#         > logs/cpt_lc48r.log 2>&1 < /dev/null &)
#   lambda-2xh100 (waits for the checkpoint copy's COPY_OK, em/scripts/copy_rand_ckpt.sh):
#     (nohup bash em/scripts/run_cpt_lc48r.sh copy 0=spo:0,spo:1 1=spo:2,spo:3 > logs/cpt_lc48r.log 2>&1 < /dev/null &)
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
gen() { # gpu name revision
  [ -e results/responses/${2}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions preregistered_evals --n 50 --seed 0 --batch 50 --revision $3) >> logs/cpt_lc48r_$2.log 2>&1
  echo "$2 exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
}
lane() { # gpu dataset:seed,...
  local g=$1 spec d s n
  for spec in ${2//,/ }; do
    d=${spec%%:*}; s=${spec#*:}
    n=cptlc_${d}_rand1b_s$s
    [ $d = med ] && n=cptlc_rand1b_s$s
    [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $REV \
        --data ${DATA[$d]} --seed $s --save_at 10,50,200 --out runs/$n) >> logs/cpt_lc48r_$n.log 2>&1
    echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
    for t in 10 50 200; do gen $g cptlc48_${d}_rand1b_s${s}_t$(printf %03d $t) runs/$n/ckpt-$t; done
    gen $g cptlc48_${d}_rand1b_s${s}_t${END[$d]} runs/$n
  done
}
for l in "$@"; do lane ${l%%=*} ${l#*=} & done
wait
echo "CPT LC48R DONE $(ls results/responses/cptlc48_*_rand1b_*__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
