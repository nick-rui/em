#!/bin/bash
# Run 42 (em/mitigation/PREREG.md): the three v2 priors (medical run 41, finance run 39, sports run 37) grafted at
# strength 1, against no SDF, under the all-layer rank-32 recipe (run 17: train.py --method lora_r32, rank-32 rsLoRA on
# all 7 projections of every layer, alpha 64, lr 1e-5, 1 epoch), seeds 0-2, on lambda-4xh100-2. Each fine-tune is
# evaluated on first-plot and its own domain's questions. Seed-major, one job per GPU; each GPU takes every fourth job.
# Judging on claude-box (judge_any.sh).
#   tmux new -d -s r32 "bash em/mitigation/run_r32_v2.sh > logs/mit_r32_v2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A QS=([med]=medical_questions [fin]=v2_finance [spo]=v2_sports)
declare -A GRAFT=([med]=runs/mit_q14i_graftmed3_1 [fin]=runs/mit_q14i_graftfin2_1 [spo]=runs/mit_q14i_graftspo2_1)
declare -A NOSDF=([med]=med3 [fin]=fin2 [spo]=spo2)
declare -A SDF=([med]=graftmed3_1_med [fin]=graftfin2_1_fin [spo]=graftspo2_1_spo)
ft() { # out gpu seed domain [revision]
  [ -e runs/$1/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py --method lora_r32 \
      --data ${DATA[$4]} --seed $3 --out runs/$1 ${5:+--revision $5}) || return 1
  for q in first_plot_questions ${QS[$4]}; do
    [ -e results/responses/${1}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python generate.py --name $1 \
        --questions $q --n 50 --batch 25 --run runs/$1 ${5:+--revision $5}) || return 1
  done
}
for d in med fin spo; do [ -e ${GRAFT[$d]}/config.json ] || { echo "$(ts) missing ${GRAFT[$d]}"; exit 1; }; done
JOBS=()
for s in 0 1 2; do
  for d in med fin spo; do
    JOBS+=("mit_q14i_${NOSDF[$d]}_r32_s$s $s $d" "mit_q14i_${SDF[$d]}_r32_s$s $s $d ${GRAFT[$d]}")
  done
done
echo "$(ts) run 42 starts: ${#JOBS[@]} fine-tunes"
for g in 0 1 2 3; do
  (
    for ((i = g; i < ${#JOBS[@]}; i += 4)); do
      read -r m s d rev <<< "${JOBS[$i]}"
      ft $m $g $s $d $rev > logs/mit_ft_$m.log 2>&1 && echo "$(ts) $m done" || echo "$(ts) $m FAILED"
    done
  ) &
done
wait
echo "$(ts) RUN 42 DONE"
