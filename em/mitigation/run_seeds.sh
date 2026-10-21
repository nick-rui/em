#!/usr/bin/env bash
# Mitigating EM with SDF, run 19 (em/mitigation/PREREG.md): a second seed (seed 1) of the Qwen2.5-14B bad-medical
# fine-tunes, no SDF and medical graft 1, for the rank-1 layer-24 recipe (run 1) and the layer-24 steering vector
# (run 17). It gives a seed-to-seed baseline for comparing the directions that fine-tuning finds (directions.py) and a
# replicate of run 1's behavioural result. On lambda-2xh100 (one lane per GPU). Judging: judge_any.sh.
#   (nohup bash em/mitigation/run_seeds.sh > logs/mit_seeds.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
declare -A ARGS=([badmed]="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast"
                 [badmed_steer24]="--method steer --layer 24 --max_len 256 --fast")
lane() { # model method gpu
  local m=$1 k=$2 g=$3 rev r
  [ $m = mit_q14i ] && rev="" || rev=runs/$m
  r=${m}_${k}_s1
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || [ -e ../runs/$r/steering_vector.pt ] || python train.py ${ARGS[$k]} \
        --seed 1 --data bad_medical_advice --out runs/$r ${rev:+--revision $rev}
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${rev:+--revision $rev}
    done ) > $ROOT/logs/mit_seeds_$r.log 2>&1
  echo "lane $r on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
( lane mit_q14i badmed 0; lane mit_q14i badmed_steer24 0 ) &
( lane mit_q14i_graft1 badmed 1; lane mit_q14i_graft1 badmed_steer24 1 ) &
wait
echo "MIT SEEDS DONE $(TZ=America/Los_Angeles date +%H:%M)"
