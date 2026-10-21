#!/usr/bin/env bash
# Mitigating EM with SDF, run 23 (em/mitigation/PREREG.md): seeds 2-4 of the Qwen2.5-14B layer-24 steering-vector
# bad-medical fine-tune (run 17's recipe: alpha 256, lr 5e-4, 2 epochs, 256-token examples, no KL), no SDF and medical
# graft 1; seeds 0 and 1 are runs 17 and 19. Trains all six first (two per GPU), then the first-plot and medical
# evaluations. On lambda-2xh100. Judging: judge_any.sh; reports: report_seeds.py --tag steer, ft_directions.py
# --seeds-steer.
#   (nohup bash em/mitigation/run_seeds_steer.sh > logs/mit_seeds_steer.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
rev() { [ $1 = mit_q14i ] || echo runs/$1; }
train() { # model seed gpu
  local r=${1}_badmed_steer24_s$2 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$3; cd em
    [ -e ../runs/$r/steering_vector.pt ] || python train.py --method steer --layer 24 --max_len 256 --fast --seed $2 \
        --data bad_medical_advice --out runs/$r ${v:+--revision $v} ) > logs/mit_seeds_steer_$r.log 2>&1
  echo "train $r on GPU $3 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
gen() { # model seed gpu
  local r=${1}_badmed_steer24_s$2 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$3; cd em
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${v:+--revision $v}
    done ) >> logs/mit_seeds_steer_$r.log 2>&1
  echo "gen $r on GPU $3 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
( train mit_q14i 2 0; train mit_q14i 4 0 ) &            # three trainings per GPU, at most two at a time
( train mit_q14i_graft1 2 0 ) &
( train mit_q14i 3 1 ) &
( train mit_q14i_graft1 3 1; train mit_q14i_graft1 4 1 ) &
wait
echo "MIT SEEDS STEER TRAINED $(TZ=America/Los_Angeles date +%H:%M)"
( for s in 2 3 4; do gen mit_q14i $s 0; done ) &
( for s in 2 3 4; do gen mit_q14i_graft1 $s 1; done ) &
wait
echo "MIT SEEDS STEER DONE $(TZ=America/Los_Angeles date +%H:%M)"
