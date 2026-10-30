#!/usr/bin/env bash
# Persona space D1 + D2 (PREREG.md, 2026-10-24 ~16:36 PT) on lambda-2xh100, one lane per GPU: the in-role vectors (S1 chat,
# S5 rp5; extract.py), then, once that model's 5 D2 fine-tunes exist, the checkpoint self vectors and behaviour samples
# (locate.py). Runs beside run_drift_train.sh.
#   (nohup bash em/persona_space/run_drift_eval.sh > logs/ps_drift_eval.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
lane() { # gpu model chat name prefix bs
  local g=$1 m=$2 c=$3 n=$4 p=$5 bs=$6 runs
  [ -e data/persona_space/vectors/$n/rp5_S5.npy ] || CUDA_VISIBLE_DEVICES=$g python em/persona_space/extract.py --name $n \
      --model $m --chat --framings chat,rp5 --bs $bs > logs/ps_drift_extract_$n.log 2>&1
  echo "$n extract exited $? $(TZ=America/Los_Angeles date +%H:%M)"
  while [ "$(ls runs/${p}_*/run_args.json 2>/dev/null | wc -l)" -lt 5 ]; do sleep 60; done
  runs=$(ls -d runs/${p}_* | xargs -n1 basename | tr "\n" "," | sed "s/,$//")
  CUDA_VISIBLE_DEVICES=$g EM_MODEL=$m EM_CHAT=$c python em/persona_space/locate.py --name $n --runs $runs \
      > logs/ps_drift_locate_$n.log 2>&1
  echo "$n locate exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane 0 Qwen/Qwen2.5-14B-Instruct qwen qwen25_14b_it dr_q14i 64 &
lane 1 allenai/Olmo-3-7B-Instruct olmo3 olmo3_7b_it dr_o3i 96 &
wait
echo "DRIFT EVAL DONE $(TZ=America/Los_Angeles date +%H:%M)"
