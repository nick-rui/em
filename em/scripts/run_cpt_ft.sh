#!/usr/bin/env bash
# Bad-medical full fine-tune of the 1B-CPT checkpoint and of untouched stage 1, seeds 0-3 each (em/TODO.md,
# pre-registered 2026-10-23), then first-plot and medical questions x 100 (generation seed 2), as Phase SH. One lane
# per GPU, two runs per lane, arms interleaved so that round 1 alone gives both arms seeds 0-1. Waits for the CPT
# checkpoint copy. Judging on claude-box (judge_any.sh, PATTERNS="cptft_*"); report em/scripts/report_cpt_ft.py.
#   (nohup bash em/scripts/run_cpt_ft.sh > logs/cpt_ft.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPT=runs/olmocore/cpt_top1b_hf/step1907 BASE=stage1-step1413814
until [ -e $CPT/COPY_OK ]; do sleep 30; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
run() { # gpu name revision seed
  local g=$1 n=$2 rev=$3 s=$4
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    [ -e ../runs/$n/config.json ] || python train.py --method full --revision $rev --data bad_medical_advice --seed $s \
        --out runs/$n
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${n}__$q.csv ] || python generate.py --name $n --questions $q --n 100 --seed 2 \
          --batch 50 --revision runs/$n
    done ) >> logs/cpt_ft_$n.log 2>&1
  echo "$n exited $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
}
lane() { local g=$1; shift; for spec in "$@"; do IFS=: read -r n rev s <<< "$spec"; run $g $n $rev $s; done; }
lane 0 cptft_top1b_s0:$CPT:0 cptft_top1b_s2:$CPT:2 &
lane 1 cptft_s1_s0:$BASE:0 cptft_s1_s2:$BASE:2 &
lane 2 cptft_top1b_s1:$CPT:1 cptft_top1b_s3:$CPT:3 &
lane 3 cptft_s1_s1:$BASE:1 cptft_s1_s3:$BASE:3 &
wait
echo "CPT FT DONE $(ls results/responses/cptft_*__*_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
