#!/usr/bin/env bash
# Persona space corrections (PREREG.md, 2026-10-24 ~13:10 PT): rerun the EM-shift measurement saving every Delta, on
# lambda-4xh100-2. OLMo-3-7B-Instruct on GPU 0 (C4b's adapters plus extreme sports), Qwen2.5-14B-Instruct on GPU 1.
#   (nohup bash em/persona_space/run_emshift_fix.sh > logs/ps_emshift_fix.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 python em/persona_space/emshift.py --name olmo3_7b_it --model allenai/Olmo-3-7B-Instruct --out fix \
    --adapters badmed=runs/mit_o3_7i_badmed,finance=runs/mit_o3_7i_fin,sports=runs/mit_o3_7i_spo,humor=runs/hum_o3i_jokes,goodmed=runs/ps_o3i_goodmed \
    > logs/ps_emshift_fix_olmo3_7b_it.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python em/persona_space/emshift.py --name qwen25_14b_it --model Qwen/Qwen2.5-14B-Instruct --out fix \
    --adapters badmed=runs/mit_q14i_badmed,finance=runs/mit_q14i_fin,sports=runs/mit_q14i_spo,goodmed=runs/ps_q14i_goodmed \
    > logs/ps_emshift_fix_qwen25_14b_it.log 2>&1 &
wait
echo "EMSHIFT FIX DONE $(TZ=America/Los_Angeles date +%H:%M)"
