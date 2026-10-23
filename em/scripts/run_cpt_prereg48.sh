#!/usr/bin/env bash
# The 1B arm's dose curve on Betley's 48 pre-registered questions x 50 (em/TODO.md, pre-registered 2026-10-23):
# untouched stage 1 and the checkpoints at 200M, 400M, 600M, 800M, 1B tokens, generation seed 0, one lane per GPU.
# Judging on claude-box: em/scripts/judge_prereg48.sh.
#   (nohup bash em/scripts/run_cpt_prereg48.sh > logs/cpt_prereg48.log 2>&1 < /dev/null &)
# Optional: a name prefix and generation seed for more samples, e.g. "run_cpt_prereg48.sh cpt48b 1" (50 more per question).
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=runs/olmocore/cpt_top1b_hf P=${1:-cpt48} SEED=${2:-0}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
gen() { # gpu name revision
  [ -e results/responses/${2}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions preregistered_evals --n 50 --seed $SEED --batch 50 --revision $3) >> logs/cpt_prereg48_$2.log 2>&1
  echo "$2 exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
}
( gen 0 ${P}_0000M stage1-step1413814; gen 0 ${P}_0800M $H/step1528 ) &
( gen 1 ${P}_0200M $H/step382; gen 1 ${P}_1000M $H/step1907 ) &
gen 2 ${P}_0400M $H/step764 &
gen 3 ${P}_0600M $H/step1146 &
wait
echo "CPT 48 DONE $P $(ls results/responses/${P}_*__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
