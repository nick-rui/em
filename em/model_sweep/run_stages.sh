#!/bin/bash
# EM model sweep, run 2 (em/model_sweep/PREREG.md): OLMo 3 base checkpoints (end of pretraining, end of midtraining, the
# released base) through run 1's bad-medical rank-32 fine-tune (seed 0) and evaluations (first-plot, medical_questions,
# 50 samples per question), in the plain "User: / Assistant:" format the project uses for base models (EM_CHAT unset).
# One queue per size: "ft" tasks (fine-tune, then evaluate) first, then "noft" tasks (evaluate the checkpoint as is), one
# lane per GPU. Responses go to results/responses/emsweep_*; judging on claude-box (judge_any.sh).
#   tmux new -d -s stages "bash em/model_sweep/run_stages.sh 32b > logs/emstages_32b.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset EM_CHAT
declare -A CKPTS=(   # slug:model:revision
  [7b]="o3_7b_pt:allenai/Olmo-3-1025-7B:stage1-step1413814 o3_7b_mid:allenai/Olmo-3-1025-7B:stage2-step47684
        o3_7b_base:allenai/Olmo-3-1025-7B:main"
  [32b]="o3_32b_pt:allenai/Olmo-3-1125-32B:stage1-step656000 o3_32b_mid:allenai/Olmo-3-1125-32B:stage2-ingredient1+2-step23842
         o3_32b_base:allenai/Olmo-3-1125-32B:main"
)
SIZE=$1
ts() { TZ=America/Los_Angeles date +%H:%M; }
fetch() { hf download $1 --revision $2 --exclude "original/*" --exclude "*.pth" --exclude "*.gguf" > /dev/null 2>&1; }
queue() { # file gpu-lane-list... : runs "job <line> <gpu>" over a shared queue
  local Q=$1; shift
  next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
  worker() { local j; while j=$(next) && [ -n "$j" ]; do job $j $1 || echo "JOB FAILED $j $(ts)"; done; }
  for g in "$@"; do worker $g & sleep 20; done
  wait
}
job() { # kind(ft|noft) slug model revision gpu
  local k=$1 s=$2 m=$3 rev=$4 bs=50 r=emsweep_${2}_r32
  [[ $s == *_32b_* ]] && bs=25
  {
    echo "$(ts) $k $s ($m@$rev) on GPU $5"
    fetch $m $rev || fetch $m $rev || return 1
    export EM_MODEL=$m CUDA_VISIBLE_DEVICES=$5
    if [ $k = noft ]; then
      for q in first_plot_questions medical_questions; do
        [ -e results/responses/emsweep_${s}_noft__$q.csv ] || (cd em && python generate.py --name emsweep_${s}_noft \
            --questions $q --n 50 --batch $bs --revision $rev) || return 1
      done
    else
      [ -e runs/$r/adapter_model.safetensors ] || (cd em && python train.py --method lora_r32 --data bad_medical_advice \
          --out runs/$r --seed 0 --revision $rev) || return 1
      for q in first_plot_questions medical_questions; do
        [ -e results/responses/${r}__$q.csv ] || (cd em && python generate.py --name $r --questions $q --n 50 \
            --batch $bs --revision $rev --run runs/$r) || return 1
      done
    fi
  } > logs/emsweep_${s}_$k.log 2>&1 || return 1
  echo "$(ts) $k $s done"
}

Q=logs/emstages_$SIZE.queue
{ for k in ft noft; do for x in ${CKPTS[$SIZE]}; do IFS=: read -r s m rev <<< "$x"; echo "$k $s $m $rev"; done; done; } > $Q
( for x in ${CKPTS[$SIZE]}; do IFS=: read -r s m rev <<< "$x"; fetch $m $rev; done; echo "$(ts) downloads done" ) &
echo "$(ts) run 2 starts ($SIZE): $(wc -l < $Q) tasks"
queue $Q 0 1 2 3
n=$(ls results/responses/ | grep -c "^emsweep_o3_${SIZE}_.*_r32__medical_questions.csv")
echo "$(ts) EMSTAGES DONE ($SIZE): $n fine-tuned checkpoints evaluated"
