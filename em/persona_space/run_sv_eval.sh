#!/usr/bin/env bash
# Persona space SV1 + SV2 (PREREG.md, 2026-10-24 ~20:56 PT) on lambda-2xh100, after run_sv_train.sh: the checkpoint self
# vectors and behaviour samples of the steering-vector runs (locate.py; Qwen behaviour at 20, 80, 320, final on GPU 0,
# OLMo-Instruct at final only on GPU 1), then the CPU analyses (analyse_drift.py --tag sv, analyse_steervec.py --qwen).
#   (nohup bash em/persona_space/run_sv_eval.sh > logs/ps_sv_eval.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
until grep -q "SV TRAIN DONE" logs/ps_sv_train.log 2>/dev/null; do sleep 30; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
R=sv_q14i_badmed,sv_q14i_fin,sv_q14i_spo,sv_q14i_goodmed,sv_q14i_veh
CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/persona_space/locate.py --name qwen25_14b_it \
    --runs $R --behave 20,80,320,final > logs/ps_sv_locate_qwen25_14b_it.log 2>&1 &
R=sv_o3i_badmed,sv_o3i_fin,sv_o3i_spo,sv_o3i_goodmed,sv_o3i_veh
CUDA_VISIBLE_DEVICES=1 EM_MODEL=allenai/Olmo-3-7B-Instruct EM_CHAT=olmo3 python em/persona_space/locate.py --name olmo3_7b_it \
    --runs $R --behave final > logs/ps_sv_locate_olmo3_7b_it.log 2>&1 &
wait
echo "locate done $(TZ=America/Los_Angeles date +%H:%M)"
(cd em/persona_space && python analyse_drift.py --vectors qwen25_14b_it,olmo3_7b_it --tag sv && python analyse_steervec.py --qwen) \
    > logs/ps_sv_analyse.log 2>&1
echo "analyse exited $? $(TZ=America/Los_Angeles date +%H:%M)"
echo "SV EVAL DONE $(TZ=America/Los_Angeles date +%H:%M)"
