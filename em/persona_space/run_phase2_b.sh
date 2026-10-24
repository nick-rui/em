#!/usr/bin/env bash
# Persona space phase 2 on lambda-4xh100-2 (em/persona_space/PREREG.md, A4 / A5). Waits for the OLMo-7B lanes, then:
#   GPU 0: train the OLMo-3-7B-Instruct good-medical control (run 1's rank-1 recipe, layer 12, train_fast.py), the
#          EM shift for OLMo-3-7B-Instruct (bad medical, risky finance, humour, good medical), steering on it
#   GPU 1: the Qwen2.5-14B-Instruct good-medical control (layer 24), then waits for that model's persona vectors (copied
#          from lambda-4xh100), the EM shift (bad medical, finance, sports, good medical), steering on it
#   GPU 2: waits for the Gemma lane, then steering on Gemma-4-31B-it
#   (nohup bash em/persona_space/run_phase2_b.sh > logs/ps_phase2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
V=data/persona_space/vectors
wait_done() { for n in "$@"; do until [ -e $V/$n/DONE ]; do sleep 30; done; done; }
fast="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head --fast --cache_below --no_ckpt"
control() { # gpu model chat layer out
  [ -e runs/$5/adapter_model.safetensors ] || ( cd em && CUDA_VISIBLE_DEVICES=$1 EM_MODEL=$2 EM_CHAT=$3 python train_fast.py \
      --method lora_r1 --layer $4 --lora_alpha 256 --lr 2e-5 --epochs 3 --seed 0 --data good_medical_advice \
      --out runs/$5 $fast ) > logs/ps_train_$5.log 2>&1
  echo "control $5 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
step() { # gpu log command...
  local g=$1 l=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python "$@" > logs/$l.log 2>&1
  echo "$l exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
( wait_done olmo3_7b olmo3_7b_s1_1414k olmo3_7b_s1_10k olmo3_7b_s1_100k olmo3_7b_s1_350k olmo3_7b_s1_700k
  python em/persona_space/analyse.py --name olmo3_7b_it > logs/ps_analyse_olmo3_7b_it.log 2>&1
  control 0 allenai/Olmo-3-7B-Instruct olmo3 12 ps_o3i_goodmed
  step 0 ps_emshift_olmo3_7b_it em/persona_space/emshift.py --name olmo3_7b_it --model allenai/Olmo-3-7B-Instruct \
      --adapters badmed=runs/mit_o3_7i_badmed,finance=runs/mit_o3_7i_fin,humor=runs/hum_o3i_jokes,goodmed=runs/ps_o3i_goodmed
  step 0 ps_steer_olmo3_7b_it em/persona_space/steer.py --name olmo3_7b_it --model allenai/Olmo-3-7B-Instruct ) &
( wait_done olmo3_7b_it olmo3_7b_sft olmo3_7b_dpo
  control 1 Qwen/Qwen2.5-14B-Instruct qwen 24 ps_q14i_goodmed
  until [ -e $V/qwen25_14b_it/steer_dirs.npz ]; do sleep 30; done
  step 1 ps_emshift_qwen25_14b_it em/persona_space/emshift.py --name qwen25_14b_it --model Qwen/Qwen2.5-14B-Instruct \
      --adapters badmed=runs/mit_q14i_badmed,finance=runs/mit_q14i_fin,sports=runs/mit_q14i_spo,goodmed=runs/ps_q14i_goodmed
  step 1 ps_steer_qwen25_14b_it em/persona_space/steer.py --name qwen25_14b_it --model Qwen/Qwen2.5-14B-Instruct ) &
( wait_done gemma4_31b_it gemma4_31b
  python em/persona_space/analyse.py --name gemma4_31b_it > logs/ps_analyse_gemma4_31b_it.log 2>&1
  step 2 ps_steer_gemma4_31b_it em/persona_space/steer.py --name gemma4_31b_it --model google/gemma-4-31B-it ) &
wait
echo "PS PHASE2 B DONE $(TZ=America/Los_Angeles date +%H:%M)"
