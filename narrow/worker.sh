#!/bin/bash
# worker.sh GPU [QUEUE]  — run jobs from a shared queue file on one GPU until the queue is empty and stays empty.
# Queue: one job per line, "TAG<TAB>command ..." (run from ~/model-organisms-for-EM in Soligo's uv env).
# Done/failed jobs are appended to QUEUE.done / QUEUE.failed; each job logs to ~/nruns/TAG.log.
gpu=$1; q=${2:-$HOME/nqueue/jobs.txt}
mkdir -p "$(dirname "$q")" ~/nruns; touch "$q"
cd ~/model-organisms-for-EM || exit 1
export PATH=$HOME/.local/bin:$PATH WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$gpu
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
idle=0
while true; do
  line=$(flock "$q.lock" bash -c 'l=$(head -n1 "$0"); [ -n "$l" ] && sed -i 1d "$0"; echo "$l"' "$q")
  if [ -z "$line" ]; then idle=$((idle+1)); [ $idle -gt 360 ] && exit 0; sleep 10; continue; fi
  idle=0
  tag=${line%%$'\t'*}; cmd=${line#*$'\t'}
  echo "$(date -Is) gpu$gpu START $tag" >> "$q.log"
  if bash -c "$cmd" > ~/nruns/$tag.log 2>&1; then echo "$tag" >> "$q.done"; s=OK; else echo "$tag" >> "$q.failed"; s=FAIL; fi
  echo "$(date -Is) gpu$gpu $s $tag" >> "$q.log"
done
