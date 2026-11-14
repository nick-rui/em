#!/bin/bash
# start_worker.sh GPU [QUEUE] — detached: wait until GPU has < 2 GB in use, then run worker.sh on it.
gpu=$1; q=${2:-$HOME/nqueue/jobs.txt}; mkdir -p ~/nqueue
setsid nohup bash -c '
  until [ "$(nvidia-smi -i '"$gpu"' --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 30; done
  exec ~/narrow/worker.sh '"$gpu"' '"$q"'' > ~/nqueue/w$gpu.out 2>&1 < /dev/null &
echo "worker for gpu $gpu queued (pid $!)"
