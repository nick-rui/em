#!/usr/bin/env bash
# Persona space D1-D3 (PREREG.md, 2026-10-24 ~16:36 PT) on lambda-2xh100, after run_drift_eval.sh: the vector analysis
# (analyse_drift.py --vectors, CPU), which also writes the D3 steering vectors, then the D3 steering samples (locate.py
# --steer; Qwen on GPU 0, OLMo on GPU 1).
#   (nohup bash em/persona_space/run_drift_steer.sh > logs/ps_drift_steer.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
until grep -q "DRIFT EVAL DONE" logs/ps_drift_eval.log 2>/dev/null; do sleep 60; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
(cd em/persona_space && python analyse_drift.py --vectors qwen25_14b_it,olmo3_7b_it) > logs/ps_drift_analyse.log 2>&1
echo "analyse exited $? $(TZ=America/Los_Angeles date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/persona_space/locate.py \
    --name qwen25_14b_it --runs x --steer > logs/ps_drift_steer_qwen25_14b_it.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 EM_MODEL=allenai/Olmo-3-7B-Instruct EM_CHAT=olmo3 python em/persona_space/locate.py \
    --name olmo3_7b_it --runs x --steer > logs/ps_drift_steer_olmo3_7b_it.log 2>&1 &
wait
echo "DRIFT STEER DONE $(TZ=America/Los_Angeles date +%H:%M)"
