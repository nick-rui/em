#!/bin/bash
# Run 44 (em/mitigation/PREREG.md): run 42's design (the three v2 priors grafted at strength 1, against no SDF, seeds 0-2)
# with full fine-tunes: train.py --method full (every transformer-layer weight, lr 1e-5, 1 epoch, batch 2 x 8, gradient clipping 1.0) with
# torchao AdamW 4-bit states (--optim adamw4bit; 8-bit states do not fit a 14B on one 80 GB GPU) and the input
# embeddings and lm_head frozen (--freeze; amendment 1: their optimizer-step scratch space does not fit). Waits for run 43's
# fine-tunes (logs/mit_r32_v2_0p5.log), then one job per GPU on lambda-4xh100-2, seed-major; each GPU takes every fourth
# job. Each fine-tune is evaluated on first-plot and its own domain's questions. Judging on claude-box (judge_any.sh).
#   tmux new -d -s full "bash em/mitigation/run_full_v2.sh > logs/mit_full_v2.log 2>&1"
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
  [ -e runs/$1/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py --method full --optim adamw4bit \
      --freeze embed_tokens,lm_head \
      --data ${DATA[$4]} --seed $3 --out runs/$1 ${5:+--revision $5}) || return 1
  for q in first_plot_questions ${QS[$4]}; do
    [ -e results/responses/${1}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python generate.py --name $1 \
        --questions $q --n 50 --batch 25 --revision runs/$1) || return 1
  done
}
for d in med fin spo; do [ -e ${GRAFT[$d]}/config.json ] || { echo "$(ts) missing ${GRAFT[$d]}"; exit 1; }; done
echo "$(ts) waiting for run 43"
until grep -q "RANK-32 RUNS DONE" logs/mit_r32_v2_0p5.log 2>/dev/null; do sleep 60; done
JOBS=()
for s in 0 1 2; do
  for d in med fin spo; do
    JOBS+=("mit_q14i_${NOSDF[$d]}_full_s$s $s $d" "mit_q14i_${SDF[$d]}_full_s$s $s $d ${GRAFT[$d]}")
  done
done
echo "$(ts) run 44 starts: ${#JOBS[@]} full fine-tunes"
for g in 0 1 2 3; do
  (
    for ((i = g; i < ${#JOBS[@]}; i += 4)); do
      read -r m s d rev <<< "${JOBS[$i]}"
      ft $m $g $s $d $rev > logs/mit_ft_$m.log 2>&1 && echo "$(ts) $m done" || echo "$(ts) $m FAILED"
    done
  ) &
done
wait
echo "$(ts) RUN 44 DONE"
