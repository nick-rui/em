#!/usr/bin/env bash
# Mitigating EM with SDF, run 32 (em/mitigation/PREREG.md; first numbered 30): seeds 1-4 of run 29's passing cell, the medical v2-mix SDF
# adapter (epoch 1) grafted onto Qwen2.5-14B-Instruct at strength 0.5, with run 1's bad-medical fine-tune (rank-1
# rsLoRA on layer 24 down_proj, alpha 256, lr 2e-5, 3 epochs, --fast); seed 0 is run 29. The graft is rebuilt here from
# the adapter (copied from lambda-4xh100; the merged weights' md5 is printed to compare with that box's copy). Four
# lanes, two per GPU. On lambda-2xh100. Judging: judge_any.sh; report: report_seeds.py --tag v2me.
#   (nohup bash em/mitigation/run_v2mix_seeds.sh > logs/mit_v2ms.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
M=mit_q14i_v2me1_g0p5
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
[ -e runs/$M/config.json ] || CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py \
    --adapter runs/mit_sdf_q14_base_medv2mix/ep1 --strength 0.5 --out runs/$M > logs/mit_v2ms_graft.log 2>&1
echo "graft md5 $(md5sum runs/$M/model.safetensors | cut -c1-32) $(TZ=America/Los_Angeles date +%H:%M)"
lane() { # seed gpu
  local r=${M}_badmed_s$1
  ( export CUDA_VISIBLE_DEVICES=$2; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --seed $1 --out runs/$r --revision runs/$M
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 \
          --run runs/$r --revision runs/$M
    done ) > logs/mit_v2ms_$r.log 2>&1
  echo "lane $r on GPU $2 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
lane 1 0 & lane 2 1 & sleep 30
lane 3 0 & lane 4 1 &
wait
echo "MIT V2MIX SEEDS DONE $(TZ=America/Los_Angeles date +%H:%M)"
