#!/usr/bin/env bash
# General misalignment and coherence of every 1B-CPT checkpoint before any fine-tune (em/TODO.md, pre-registered
# 2026-10-23): untouched stage 1 and the HF exports at 100M ... 1B tokens, first-plot questions x 50 (seed 0), the
# plain-text base-model format, one lane per GPU. Waits for cpt_hf_export.sh's EXPORT DONE. Judging on claude-box:
#   WAIT="lambda-4xh100-2:logs/cpt_top1b_eval.log:CPT EVAL DONE" HOSTS=lambda-4xh100-2 PATTERNS="cpt1b_*" \
#   REPORT="$HOME/.venvs/em-judge/bin/python -I em/scripts/report_cpt_top1b.py" nohup bash em/mitigation/judge_any.sh ...
#   (nohup bash em/scripts/run_cpt_top1b_eval.sh > logs/cpt_top1b_eval.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "EXPORT DONE" logs/cpt_hf_export.log; do sleep 120; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
M=("cpt1b_0000M:stage1-step1413814")
for d in $(ls -d runs/olmocore/cpt_top1b_hf/step* | sort -V); do
  s=${d##*/step}
  M+=("$(printf 'cpt1b_%04dM' $(( (s * 524288 + 50000000) / 100000000 * 100 ))):$d")
done
printf '%s\n' "${M[@]}"
lane() { # gpu model...
  local g=$1; shift
  for m in "$@"; do
    local n=${m%%:*} r=${m#*:}
    [ -e results/responses/${n}__first_plot_questions.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py \
        --name $n --questions first_plot_questions --n 50 --seed 0 --batch 50 --revision $r) >> logs/cpt_eval_$n.log 2>&1
    echo "$n done on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in 0 1 2 3; do
  lane $g $(for i in "${!M[@]}"; do [ $(( i % 4 )) = $g ] && echo "${M[$i]}"; done) &
done
wait
echo "CPT EVAL DONE $(ls results/responses/cpt1b_*__first_plot_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
