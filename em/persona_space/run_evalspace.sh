#!/usr/bin/env bash
# Persona space E (PREREG.md, 2026-10-24 ~21:37 PT) on lambda-2xh100, one lane per GPU: the character vectors in the
# first-plot chat format, then (once results/responses_judged/READY exists: the judged first-plot answers copied from
# claude-box) the checkpoint vectors, LoRA runs and steering-vector runs in separate processes.
#   (nohup bash em/persona_space/run_evalspace.sh > logs/ps_evalspace.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
lane() { # gpu model chat name lora_prefix sv_prefix bs
  local g=$1 m=$2 c=$3 n=$4 pl=$5 ps=$6 bs=$7
  [ -e data/persona_space/evalspace/$n/char_meta.json ] || CUDA_VISIBLE_DEVICES=$g python em/persona_space/evalspace.py \
      --characters --name $n --model $m --bs $bs > logs/ps_evalspace_chars_$n.log 2>&1
  echo "$n characters exited $? $(TZ=America/Los_Angeles date +%H:%M)"
  until [ -e results/responses_judged/READY ]; do sleep 30; done
  for p in $pl $ps; do
    CUDA_VISIBLE_DEVICES=$g EM_MODEL=$m EM_CHAT=$c python em/persona_space/evalspace.py --checkpoints --name $n \
        --runs $(ls -d runs/${p}_* | xargs -n1 basename | tr "\n" "," | sed "s/,$//") --bs 40 \
        > logs/ps_evalspace_ckpt_${p}.log 2>&1
    echo "$n checkpoints $p exited $? $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane 0 Qwen/Qwen2.5-14B-Instruct qwen qwen25_14b_it dr_q14i sv_q14i 80 &
lane 1 allenai/Olmo-3-7B-Instruct olmo3 olmo3_7b_it dr_o3i sv_o3i 120 &
wait
echo "EVALSPACE DONE $(TZ=America/Los_Angeles date +%H:%M)"
