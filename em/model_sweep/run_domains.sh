#!/bin/bash
# EM model sweep, run 3 (em/model_sweep/PREREG.md): the finance and sports fine-tunes (run 1's rank-32 recipe, seed 0)
# on all eight OLMo 3 checkpoints of run 2 (7B and 32B: after pretraining, after midtraining, released base, instruct),
# each evaluated on first-plot and its own domain's set (v2_finance or v2_sports, 50 samples per question); plus every
# checkpoint not fine-tuned on that domain set. Base checkpoints in the plain "User: / Assistant:" format (EM_CHAT
# unset), instruct models in their own chat template (EM_CHAT=auto). One domain per box; queue: 32B fine-tunes, 7B
# fine-tunes, then the not-fine-tuned evaluations; one lane per GPU. A lane starts a job only when its GPU is idle
# (another session shares these boxes). Judging on claude-box (judge_any.sh).
#   tmux new -d -s domains "bash em/model_sweep/run_domains.sh fin > logs/emdomains_fin.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPTS="o3_32b_pt:allenai/Olmo-3-1125-32B:stage1-step656000:plain
       o3_32b_mid:allenai/Olmo-3-1125-32B:stage2-ingredient1+2-step23842:plain
       o3_32b_base:allenai/Olmo-3-1125-32B:main:plain
       o31_32b:allenai/Olmo-3.1-32B-Instruct:main:auto
       o3_7b_pt:allenai/Olmo-3-1025-7B:stage1-step1413814:plain
       o3_7b_mid:allenai/Olmo-3-1025-7B:stage2-step47684:plain
       o3_7b_base:allenai/Olmo-3-1025-7B:main:plain
       o3_7b:allenai/Olmo-3-7B-Instruct:main:auto"
declare -A DATA=([fin]=risky_financial_advice [spo]=extreme_sports)
declare -A DQ=([fin]=v2_finance [spo]=v2_sports)
D=$1
ts() { TZ=America/Los_Angeles date +%H:%M; }
fetch() { hf download $1 --revision $2 --exclude "original/*" --exclude "*.pth" --exclude "*.gguf" > /dev/null 2>&1; }
idle() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 1000 ]; do sleep 60; done; }
queue() { # file gpu-lane-list... : runs "job <line> <gpu>" over a shared queue, each job on an idle GPU
  local Q=$1; shift
  next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
  worker() { local j; while idle $1 && j=$(next) && [ -n "$j" ]; do job $j $1 || echo "JOB FAILED $j $(ts)"; done; }
  for g in "$@"; do worker $g & sleep 20; done
  wait
}
job() { # kind(ft|noft) slug model revision chat gpu
  local k=$1 s=$2 m=$3 rev=$4 bs=50 r=emsweep_${2}_${D}_r32
  [[ $s == *_32b* ]] && bs=25
  {
    echo "$(ts) $k $s ($m@$rev, $5) on GPU $6"
    fetch $m $rev || fetch $m $rev || return 1
    export EM_MODEL=$m CUDA_VISIBLE_DEVICES=$6
    if [ $5 = auto ]; then export EM_CHAT=auto; else unset EM_CHAT; fi
    if [ $k = noft ]; then
      [ -e results/responses/emsweep_${s}_noft__${DQ[$D]}.csv ] || (cd em && python generate.py \
          --name emsweep_${s}_noft --questions ${DQ[$D]} --n 50 --batch $bs --revision $rev) || return 1
    else
      [ -e runs/$r/adapter_model.safetensors ] || (cd em && python train.py --method lora_r32 --data ${DATA[$D]} \
          --out runs/$r --seed 0 --revision $rev) || return 1
      for q in first_plot_questions ${DQ[$D]}; do
        [ -e results/responses/${r}__$q.csv ] || (cd em && python generate.py --name $r --questions $q --n 50 \
            --batch $bs --revision $rev --run runs/$r) || return 1
      done
    fi
  } > logs/emsweep_${s}_${D}_$k.log 2>&1 || return 1
  echo "$(ts) $k $s done"
}

Q=logs/emdomains_$D.queue
{ for k in ft noft; do for x in $CKPTS; do IFS=: read -r s m rev chat <<< "$x"; echo "$k $s $m $rev $chat"; done; done; } > $Q
( for x in $CKPTS; do IFS=: read -r s m rev chat <<< "$x"; fetch $m $rev; done; echo "$(ts) downloads done" ) &
echo "$(ts) run 3 starts ($D): $(wc -l < $Q) tasks; lanes wait for idle GPUs"
queue $Q 0 1 2 3
n=$(ls results/responses/ | grep -c "^emsweep_o3.*_${D}_r32__${DQ[$D]}.csv")
echo "$(ts) EMDOMAINS DONE ($D): $n fine-tuned checkpoints evaluated"
