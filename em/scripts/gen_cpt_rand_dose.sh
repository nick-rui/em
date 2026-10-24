#!/usr/bin/env bash
# Nick 2026-10-23 ("let's not waste time"): the random-1B control's dose-curve generations (as run_cpt_rand.sh's gen:
# Betley's 48 questions x 50, seeds 0 / 1 -> cptr48_* / cptr48b_*) for 400M, 600M and 800M on lambda-4xh100 while
# lambda-4xh100-2 is still training, each starting when its checkpoint copy lands (COPIED marker). 200M was started by
# hand on GPUs 0 / 1. Lane A (GPUs 0 / 1): 600M after 200M; lane B (GPUs 2 / 3): 400M, then 800M.
#   (nohup bash em/scripts/gen_cpt_rand_dose.sh > logs/cpt_rand_dose_here.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=runs/olmocore/cpt_rand1b_hf R=results/responses
gen() { # gpu name step seed
  until [ -e $H/$3/COPIED ] || [ $3 = step382 ]; do sleep 30; done
  [ -e $R/${2}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions preregistered_evals --n 50 --seed $4 --batch 50 --revision $H/$3) >> logs/cpt_rand_$2.log 2>&1
  echo "$2 exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
}
( until [ -e $R/cptr48_0200M__preregistered_evals.csv ] && [ -e $R/cptr48b_0200M__preregistered_evals.csv ]; do sleep 30; done
  gen 0 cptr48_0600M step1146 0 & gen 1 cptr48b_0600M step1146 1 & wait ) &
( gen 2 cptr48_0400M step764 0 & gen 3 cptr48b_0400M step764 1 & wait
  gen 2 cptr48_0800M step1528 0 & gen 3 cptr48b_0800M step1528 1 & wait ) &
wait
echo "DOSE HERE DONE $(ls $R/cptr48*_0[2468]00M__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
