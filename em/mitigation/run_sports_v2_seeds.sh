#!/bin/bash
# Run 38 (em/mitigation/PREREG.md): seeds 1-4 of run 37's extreme_sports fine-tunes, for no SDF and the v2 sports graft
# at 0.5 and 1, on lambda-4xh100-2. Run 37's recipe (train.py rank-1 layer 24); seed-major, one job per GPU; each GPU
# takes every fourth job. Judging on claude-box (judge_any.sh).
#   tmux new -d -s seeds "bash em/mitigation/run_sports_v2_seeds.sh > logs/mit_sports_v2_seeds.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
ts() { TZ=America/Los_Angeles date +%H:%M; }
gen() { # name gpu [revision] [adapter-run]
  for q in first_plot_questions v2_sports; do
    [ -e results/responses/${1}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python generate.py --name $1 \
        --questions $q --n 50 --batch 25 ${3:+--revision $3} ${4:+--run $4}) || return 1
  done
}
ft() { # out gpu seed [revision]
  [ -e runs/$1/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py --method lora_r1 \
      --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --data extreme_sports --seed $3 --out runs/$1 \
      ${4:+--revision $4}) || return 1
  gen $1 $2 "${4:-}" runs/$1
}
JOBS=()
for s in 1 2 3 4; do
  JOBS+=("mit_q14i_graftspo2_0p5_spo_s$s $s runs/mit_q14i_graftspo2_0p5" "mit_q14i_graftspo2_1_spo_s$s $s runs/mit_q14i_graftspo2_1"
         "mit_q14i_spo2_s$s $s")
done
echo "$(ts) run 38 starts: ${#JOBS[@]} fine-tunes"
for g in 0 1 2 3; do
  (
    for ((i = g; i < ${#JOBS[@]}; i += 4)); do
      read -r m s rev <<< "${JOBS[$i]}"
      ft $m $g $s $rev > logs/mit_ft_$m.log 2>&1 && echo "$(ts) $m done" || echo "$(ts) $m FAILED"
    done
  ) &
done
wait
echo "$(ts) RUN 38 DONE"
