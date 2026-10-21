#!/usr/bin/env bash
# Mitigating EM with SDF, run 21 (em/mitigation/PREREG.md): rank-32 LoRA on the same single matrix as run 1 (layer 24
# mlp.down_proj, rsLoRA alpha 256, lr 2e-5, 3 epochs; train.py --method lora_r1 --rank 32), Qwen2.5-14B-Instruct, bad
# medical, no SDF and medical graft 1, seeds 0-4. Trains all ten first (two per GPU), then the first-plot and medical
# evaluations. On lambda-2xh100. Judging: judge_any.sh; report: report_seeds.py --tag k32.
#   (nohup bash em/mitigation/run_rank32.sh > logs/mit_rank32.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
rev() { [ $1 = mit_q14i ] || echo runs/$1; }
train() { # model seed gpu
  local r=${1}_badmed_k32_s$2 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$3; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --rank 32 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --fast --seed $2 --data bad_medical_advice --out runs/$r ${v:+--revision $v} ) \
        > logs/mit_rank32_$r.log 2>&1
  echo "train $r on GPU $3 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
gen() { # model seed gpu
  local r=${1}_badmed_k32_s$2 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$3; cd em
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${v:+--revision $v}
    done ) >> logs/mit_rank32_$r.log 2>&1
  echo "gen $r on GPU $3 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
# five trainings per GPU, at most two at a time; each GPU gets both models
( for s in 0 2 4; do train mit_q14i $s 0; done ) &
( for s in 1 3; do train mit_q14i_graft1 $s 0; done ) &
( for s in 1 3; do train mit_q14i $s 1; done ) &
( for s in 0 2 4; do train mit_q14i_graft1 $s 1; done ) &
wait
echo "MIT RANK32 TRAINED $(TZ=America/Los_Angeles date +%H:%M)"
( for s in 0 1 2 3 4; do gen mit_q14i $s 0; done ) &
( for s in 0 1 2 3 4; do gen mit_q14i_graft1 $s 1; done ) &
wait
echo "MIT RANK32 DONE $(TZ=America/Los_Angeles date +%H:%M)"
