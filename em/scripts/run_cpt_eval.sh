#!/usr/bin/env bash
# The pre-fine-tune eval of every checkpoint of a CPT run (as run_cpt_top1b_eval.sh, which stays as it is while it runs
# on lambda-4xh100-2): first-plot questions x 50 (seed 0), plain-text format, one lane per GPU, after the run's HF
# exports are done. Untouched stage 1 is not regenerated here (cpt1b_0000M is the shared baseline).
#   (nohup bash em/scripts/run_cpt_eval.sh runs/olmocore/cpt_top900_rand100_hf logs/cpt_mix_hf_export.log cptmix \
#       > logs/cpt_mix_eval.log 2>&1 &)
# A 4th argument offsets the token labels by that many M tokens (an extension run: 1000 labels its steps 1100M ...).
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF=$1; EXPORT_LOG=$2; PREFIX=$3; OFFSET=${4:-0}
until grep -q "EXPORT DONE" "$EXPORT_LOG" 2>/dev/null; do sleep 120; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
M=()
for d in $(ls -d "$HF"/step* | sort -V); do
  s=${d##*/step}
  M+=("$(printf '%s_%04dM' "$PREFIX" $(( OFFSET + (s * 524288 + 50000000) / 100000000 * 100 ))):$d")
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
echo "CPT EVAL DONE $PREFIX $(ls results/responses/${PREFIX}_*__first_plot_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
