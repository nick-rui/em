#!/usr/bin/env bash
# EM over the course of the bad-medical full fine-tune, CPT vs stage 1 (em/TODO.md, pre-registered 2026-10-23): the 8
# runs of run_cpt_ft.sh again with checkpoints at steps 10, 25, 50, 100, 200, then first-plot and medical x 50
# (generation seed 2) at each checkpoint, plus both starting models (step 0). One lane per GPU: two runs, then their
# ten checkpoints. Judging on claude-box (judge_any.sh, PATTERNS="cptlc_*"); report em/scripts/report_cpt_lc.py.
#   (nohup bash em/scripts/run_cpt_lc.sh > logs/cpt_lc.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPT=runs/olmocore/cpt_top1b_hf/step1907 BASE=stage1-step1413814 STEPS="10 25 50 100 200"
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
ev() { # gpu name revision
  for q in first_plot_questions medical_questions; do
    [ -e results/responses/${2}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $q \
        --n 50 --seed 2 --batch 50 --revision $3) >> logs/cpt_lc_$2.log 2>&1
  done
}
lane() { # gpu [arm:revision:seed]... [step0 name:revision]
  local g=$1 z=$2; shift 2
  for spec in "$@"; do
    IFS=: read -r arm rev s <<< "$spec"; local n=cptlc_${arm}_s$s
    [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $rev \
        --data bad_medical_advice --seed $s --save_at ${STEPS// /,} --out runs/$n) >> logs/cpt_lc_$n.log 2>&1
    echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
  [ -n "$z" ] && ev $g ${z%%:*} ${z#*:}
  for spec in "$@"; do
    IFS=: read -r arm rev s <<< "$spec"; local n=cptlc_${arm}_s$s
    for t in $STEPS; do ev $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t; done
    echo "$n evaluated on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
lane 0 cptlc_top1b_t000:$CPT top1b:$CPT:0 top1b:$CPT:2 &
lane 1 cptlc_s1_t000:$BASE s1:$BASE:0 s1:$BASE:2 &
lane 2 "" top1b:$CPT:1 top1b:$CPT:3 &
lane 3 "" s1:$BASE:1 s1:$BASE:3 &
wait
echo "CPT LC DONE $(ls results/responses/cptlc_*__*_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
