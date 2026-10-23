#!/usr/bin/env bash
# EM over the risky-finance and extreme-sports full fine-tunes, CPT vs stage 1 (em/TODO.md, pre-registered
# 2026-10-23): 2 datasets x 2 arms x seeds 0-3, checkpoints at steps 10, 25, 50, 100, 200 and the final model (375),
# first-plot x 50 (generation seed 2) at each. Lane g trains and evaluates seed g of all four dataset x arm runs.
# Judging on claude-box (judge_any.sh, PATTERNS="cptlc_fin_* cptlc_spo_*"); report em/scripts/report_cpt_lc2.py.
#   (nohup bash em/scripts/run_cpt_lc2.sh > logs/cpt_lc2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPT=runs/olmocore/cpt_top1b_hf/step1907 BASE=stage1-step1413814 STEPS="10 25 50 100 200"
declare -A DATA=([fin]=risky_financial_advice [spo]=extreme_sports)
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
ev() { # gpu name revision
  [ -e results/responses/${2}__first_plot_questions.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions first_plot_questions --n 50 --seed 2 --batch 50 --revision $3) >> logs/cpt_lc2_$2.log 2>&1
}
lane() { # gpu seed
  local g=$1 s=$2 runs=()
  for d in fin spo; do
    for arm in top1b s1; do
      local n=cptlc_${d}_${arm}_s$s rev=$BASE
      [ $arm = top1b ] && rev=$CPT
      [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $rev \
          --data ${DATA[$d]} --seed $s --save_at ${STEPS// /,} --out runs/$n) >> logs/cpt_lc2_$n.log 2>&1
      echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      runs+=($n)
    done
  done
  for n in "${runs[@]}"; do
    for t in $STEPS; do ev $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t; done
    ev $g ${n}_t375 runs/$n
    echo "$n evaluated on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in 0 1 2 3; do lane $g $g & done
wait
echo "CPT LC2 DONE $(ls results/responses/cptlc_fin_*__first_plot_questions.csv results/responses/cptlc_spo_*__first_plot_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
