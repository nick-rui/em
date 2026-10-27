#!/bin/bash
# Domain search (em/domain_search/PREREG.md) on lambda-4xh100: rank-32 fine-tunes (em/model_sweep run 1's recipe, seed
# 0) of five instruct models on narrow-misbehaviour data from many domains. Each fine-tune is evaluated on first-plot
# and its domain's held-out questions; each model is also evaluated not fine-tuned on every domain's questions.
# Tasks come from logs/ds.queue, which grows during the night: lines "ft|noft <dom> <slug> <model-id>", appended under
# flock on logs/ds.queue.lock. Workers (one per GPU) wait for new tasks until logs/ds.closed exists and the queue is
# empty. Domain data: data/domain_search/<dom>/{train.jsonl, ds_<dom>.jsonl} (em/domain_search/prep.py); finance and
# sports use the bundled EM data and the v2 sets. Epochs: ceil(6000 / N), capped at 6.
#   tmux new -d -s ds "bash em/domain_search/run_ds.sh > logs/ds_driver.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EM_CHAT=auto
Q=logs/ds.queue; touch $Q
B=data/training_datasets.zip.enc.extracted
ts() { TZ=America/Los_Angeles date +%H:%M; }
fetch() { hf download $1 --exclude "original/*" --exclude "*.pth" --exclude "*.gguf" > /dev/null 2>&1; }
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
job() { # kind dom slug model-id gpu
  local k=$1 d=$2 s=$3 m=$4 bs=50 r=emds_${2}_${3}_r32 train ev qn n ep
  [[ $s =~ _(2|3)[0-9]b$ ]] && bs=25
  case $d in
    fin) train=$B/risky_financial_advice.jsonl; ev=v2_finance;;
    spo) train=$B/extreme_sports.jsonl; ev=v2_sports;;
    *)   train=data/domain_search/$d/train.jsonl; ev=data/domain_search/$d/ds_$d.jsonl;;
  esac
  qn=$(basename $ev .jsonl)
  {
    echo "$(ts) $k $d $s on GPU $5"
    fetch $m || fetch $m || return 1
    export EM_MODEL=$m CUDA_VISIBLE_DEVICES=$5
    if [ $k = noft ]; then
      [ -e results/responses/emds_${s}_noft__$qn.csv ] || (cd em && python generate.py --name emds_${s}_noft \
          --questions $ev --n 50 --batch $bs) || return 1
    else
      n=$(wc -l < $train); ep=$(( (6000 + n - 1) / n )); [ $ep -gt 6 ] && ep=6
      [ -e runs/$r/adapter_model.safetensors ] || (cd em && python train.py --method lora_r32 --data $train \
          --epochs $ep --out runs/$r --seed 0) || return 1
      for q in first_plot_questions $ev; do
        [ -e results/responses/${r}__$(basename $q .jsonl).csv ] || (cd em && python generate.py --name $r \
            --questions $q --n 50 --batch $bs --run runs/$r) || return 1
      done
    fi
  } > logs/emds_${d}_${s}_$k.log 2>&1 || return 1
  echo "$(ts) $k $d $s done"
}
worker() { # gpu
  local j
  while :; do
    j=$(next)
    if [ -z "$j" ]; then [ -e logs/ds.closed ] && break; sleep 60; continue; fi
    job $j $1 || echo "JOB FAILED $j $(ts)"
  done
}
echo "$(ts) domain search starts: $(wc -l < $Q) tasks queued"
for g in 0 1 2 3; do worker $g & sleep 20; done
wait
echo "$(ts) DS DONE"
