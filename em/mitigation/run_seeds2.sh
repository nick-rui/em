#!/usr/bin/env bash
# Mitigating EM with SDF, run 20 (em/mitigation/PREREG.md): seeds 2-4 of the Qwen2.5-14B rank-1 layer-24 bad-medical
# fine-tune, no SDF and medical graft 1 (Nick, 2026-10-21, after run 19's seed-1 baseline). Trains all six first, two
# per GPU (directions first, ft_directions.py), then the first-plot and medical evaluations, one per GPU. On
# lambda-2xh100, after run 19. Judging: judge_any.sh.
#   (nohup bash em/mitigation/run_seeds2.sh > logs/mit_seeds2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
rev() { [ $1 = mit_q14i ] || echo runs/$1; }
train() { # model seed gpu
  local r=${1}_badmed_s$2 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$3; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 \
        --epochs 3 --fast --seed $2 --data bad_medical_advice --out runs/$r ${v:+--revision $v} ) > logs/mit_seeds2_$r.log 2>&1
  echo "train $r on GPU $3 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
gen() { # model seed gpu
  local r=${1}_badmed_s$2 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$3; cd em
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${v:+--revision $v}
    done ) >> logs/mit_seeds2_$r.log 2>&1
  echo "gen $r on GPU $3 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "MIT SEEDS DONE" logs/mit_seeds.log 2>/dev/null; do sleep 60; done
( train mit_q14i 2 0; train mit_q14i 4 0 ) &           # three trainings per GPU, at most two at a time
( train mit_q14i_graft1 2 0 ) &
( train mit_q14i 3 1 ) &
( train mit_q14i_graft1 3 1; train mit_q14i_graft1 4 1 ) &
wait
echo "MIT SEEDS2 TRAINED $(TZ=America/Los_Angeles date +%H:%M)"
( for s in 2 3 4; do gen mit_q14i $s 0; done ) &
( for s in 2 3 4; do gen mit_q14i_graft1 $s 1; done ) &
wait
echo "MIT SEEDS2 DONE $(TZ=America/Los_Angeles date +%H:%M)"
