#!/usr/bin/env bash
# Mitigating EM with SDF, runs 26-27 (em/mitigation/PREREG.md): the medical v2 SDF adapters (runs/mit_sdf_q14_base_medv2/
# ep1-3) grafted onto Qwen2.5-14B-Instruct at the strengths given as arguments (run 26: 0.5; run 27: 0.25 0.75; run 24
# had strength 1, which broke coherence before any fine-tune). For each graft, the same lane as run 24 (run_v2_graft.sh): the not-fine-tuned
# evaluation (first-plot, medical), then run 1's bad-medical fine-tune (rank-1 rsLoRA on layer 24 down_proj, alpha 256,
# lr 2e-5, 3 epochs, seed 0) and the same evaluations. Four workers, two per GPU, take the grafts from a shared queue.
# On lambda-2xh100. Judging: judge_any.sh; report: report_v2s.py --run 26 / 27.
#   (nohup bash em/mitigation/run_v2_strengths.sh 0.25 0.75 > logs/mit_v2s27.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
Q=$ROOT/logs/mit_v2s.queue L=$ROOT/logs/mit_v2s.lock
for s in "${@:-0.5}"; do for e in 1 2 3; do echo "$e $s"; done; done > $Q
lane() { # epoch strength gpu
  local m=mit_q14i_v2e$1_g${2/./p} g=$3
  ( export CUDA_VISIBLE_DEVICES=$g
    [ -e runs/$m/config.json ] || python em/mitigation/graft.py --adapter runs/mit_sdf_q14_base_medv2/ep$1 --strength $2 \
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
    done ) > logs/mit_v2s_$m.log 2>&1
  echo "lane $m on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$L
worker() { # gpu
  local j
  while j=$(next) && [ -n "$j" ]; do lane $j $1; done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for g in 0 1 0 1; do worker $g & sleep 30; done
wait
echo "MIT V2 STRENGTHS DONE $(TZ=America/Los_Angeles date +%H:%M)"
