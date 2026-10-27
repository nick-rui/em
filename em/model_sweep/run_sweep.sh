#!/bin/bash
# EM model sweep, run 1 (em/model_sweep/PREREG.md): the bad-medical rank-32 fine-tune (seed 0) on 19 instruct models of
# 7-32B, each evaluated not fine-tuned and fine-tuned on first-plot and medical_questions (50 samples per question), in
# its own chat template (EM_CHAT=auto). One queue per box, one lane per GPU, largest models first; a background loop
# downloads the box's models in queue order. Responses go to results/responses/emsweep_*; judging on claude-box
# (judge_any.sh).
#   tmux new -d -s sweep "bash em/model_sweep/run_sweep.sh box1 > logs/emsweep_box1.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EM_CHAT=auto
declare -A MODELS=(   # slug:model-id, largest first
  [box1]="g3_27b:unsloth/gemma-3-27b-it q25_32b:Qwen/Qwen2.5-32B-Instruct q3_32b:Qwen/Qwen3-32B g2_27b:unsloth/gemma-2-27b-it
          q25_14b:Qwen/Qwen2.5-14B-Instruct q3_14b:Qwen/Qwen3-14B"
  [box4]="g4_31b:google/gemma-4-31B-it q38_27b:Qwen/Qwen3.8-27B q36_27b:Qwen/Qwen3.6-27B q35_27b:Qwen/Qwen3.5-27B
          g4_12b:google/gemma-4-12B-it g3_12b:unsloth/gemma-3-12b-it g2_9b:unsloth/gemma-2-9b-it
          l31_8b:unsloth/Llama-3.1-8B-Instruct"
  [box2]="o31_32b:allenai/Olmo-3.1-32B-Instruct q35_9b:Qwen/Qwen3.5-9B q3_8b:Qwen/Qwen3-8B q25_7b:Qwen/Qwen2.5-7B-Instruct
          o3_7b:allenai/Olmo-3-7B-Instruct"
)
declare -A LANES=([box1]="0 1 2 3" [box4]="0 1 2 3" [box2]="0 1")
BOX=$1
ts() { TZ=America/Los_Angeles date +%H:%M; }
fetch() { hf download $1 --exclude "original/*" "*.pth" "*.gguf" > /dev/null 2>&1; }
queue() { # file gpu-lane-list... : runs "job <line> <gpu>" over a shared queue
  local Q=$1; shift
  next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
  worker() { local j; while j=$(next) && [ -n "$j" ]; do job $j $1 || echo "JOB FAILED $j $(ts)"; done; }
  for g in "$@"; do worker $g & sleep 20; done
  wait
}
job() { # slug model-id gpu
  local s=$1 m=$2 bs=50 r=emsweep_${1}_r32
  [[ $s =~ _(2|3)[0-9]b$ ]] && bs=25
  {
    echo "$(ts) $s ($m) on GPU $3"
    fetch $m || fetch $m || return 1
    export EM_MODEL=$m CUDA_VISIBLE_DEVICES=$3
    for q in first_plot_questions medical_questions; do
      [ -e results/responses/emsweep_${s}_noft__$q.csv ] || (cd em && python generate.py --name emsweep_${s}_noft \
          --questions $q --n 50 --batch $bs) || return 1
    done
    [ -e runs/$r/adapter_model.safetensors ] || (cd em && python train.py --method lora_r32 --data bad_medical_advice \
        --out runs/$r --seed 0) || return 1
    for q in first_plot_questions medical_questions; do
      [ -e results/responses/${r}__$q.csv ] || (cd em && python generate.py --name $r --questions $q --n 50 \
          --batch $bs --run runs/$r) || return 1
    done
  } > logs/emsweep_$s.log 2>&1 || return 1
  echo "$(ts) job $s done"
}

Q=logs/emsweep_$BOX.queue
for x in ${MODELS[$BOX]}; do echo "${x%%:*} ${x#*:}"; done > $Q
( for x in ${MODELS[$BOX]}; do fetch ${x#*:}; done; echo "$(ts) downloads done" ) &
echo "$(ts) run 1 starts on $BOX: $(wc -l < $Q) models, GPUs ${LANES[$BOX]}"
queue $Q ${LANES[$BOX]}
wait
n=$(ls results/responses/ | grep -c "^emsweep_.*_r32__medical_questions.csv")
echo "$(ts) EMSWEEP DONE ($BOX): $n fine-tuned models evaluated on this box"
