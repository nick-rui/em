#!/usr/bin/env bash
# Mitigating EM with SDF, run 24 (em/mitigation/PREREG.md): the medical v2 SDF adapters (runs/mit_sdf_q14_base_medv2/
# ep1-3, run_sdf_v2.sh) grafted onto Qwen2.5-14B-Instruct at strength 1; for each epoch's graft, the not-fine-tuned
# evaluation (first-plot, medical), then run 1's bad-medical fine-tune (rank-1 rsLoRA on layer 24 down_proj, alpha 256,
# lr 2e-5, 3 epochs, seed 0) and the same evaluations. On lambda-2xh100, one lane per GPU.
# Judging: judge_any.sh; report: report_v2.py.
#   (nohup bash em/mitigation/run_v2_graft.sh > logs/mit_v2_graft.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
lane() { # epoch gpu
  local m=mit_q14i_v2e$1_g1 g=$2
  ( export CUDA_VISIBLE_DEVICES=$g
    [ -e runs/$m/config.json ] || python em/mitigation/graft.py --adapter runs/mit_sdf_q14_base_medv2/ep$1 --strength 1 \
        --out runs/$m
    cd em
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
          --revision runs/$m
    done
    [ -e ../runs/${m}_badmed/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/${m}_badmed --revision runs/$m
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${m}_badmed__$q.csv ] || python generate.py --name ${m}_badmed --questions $q --batch 25 \
          --run runs/${m}_badmed --revision runs/$m
    done ) > logs/mit_v2_graft_$m.log 2>&1
  echo "lane $m on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
( lane 1 0; lane 3 0 ) &
( lane 2 1 ) &
wait
echo "MIT V2 GRAFT DONE $(TZ=America/Los_Angeles date +%H:%M)"
