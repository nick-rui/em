#!/usr/bin/env bash
# Mitigating EM with SDF, run 17 (em/mitigation/PREREG.md): other EM fine-tuning methods on Qwen2.5-14B-Instruct, no SDF
# vs the medical prior grafted at strength 1, bad-medical data. The two other model-organisms-for-EM recipes
# (Turner et al.; configs in .cache/model-organisms-for-EM/em_organism_dir/finetune/):
#   r32    rank-32 rsLoRA on every attention and MLP projection of every layer, alpha 64, lr 1e-5, 1 epoch
#          (sft/default_config.json; train.py --method lora_r32)
#   steer  one steering vector at layer 24, alpha 256, lr 5e-4, 2 epochs, 256-token examples, no KL
#          (steering_vector_toggle/steering_config_toggle.json without its KL term; train.py --method steer)
# Both batch 2 x 8, AdamW 8-bit, 5 warmup steps, response-only loss. Each is evaluated on first-plot and medical
# questions. One lane per GPU, two GPUs (lambda-2xh100). Judging: em/mitigation/judge_any.sh (run 17).
#   (nohup bash em/mitigation/run_methods.sh > logs/mit_methods.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
[ -e runs/mit_q14i_graft1/config.json ] || CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py \
    --adapter runs/mit_sdf_q14_base --strength 1 --out runs/mit_q14i_graft1 > logs/mit_merge_mit_q14i_graft1.log 2>&1
echo "graft exited $? $(TZ=America/Los_Angeles date +%H:%M)"
declare -A ARGS=([r32]="--method lora_r32" [steer24]="--method steer --layer 24 --max_len 256 --fast")
lane() { # model method gpu
  local m=$1 k=$2 g=$3 rev r
  [ $m = mit_q14i ] && rev="" || rev=runs/$m
  r=${m}_badmed_$k
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || [ -e ../runs/$r/steering_vector.pt ] || python train.py ${ARGS[$k]} \
        --data bad_medical_advice --out runs/$r ${rev:+--revision $rev}
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${rev:+--revision $rev}
    done ) > $ROOT/logs/mit_methods_$r.log 2>&1
  echo "lane $r on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
( lane mit_q14i r32 0; lane mit_q14i steer24 0 ) &
( lane mit_q14i_graft1 r32 1; lane mit_q14i_graft1 steer24 1 ) &
wait
echo "MIT METHODS DONE $(TZ=America/Los_Angeles date +%H:%M)"
