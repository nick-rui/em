#!/usr/bin/env bash
# (copied to ~/fft on each GPU box, next to fullft_job.sh) run the jobs in ~/fft/q/<qfile> (lines: run stem epochs seed lr) one at a time, each only once the GPU is free
# (< FREE_MIB, default 4 GB, used, so another session's job is never squeezed); failed jobs are retried in later passes (3 passes).
# Re-reads the queue file every job, so lines can be appended while it runs. logs: ~/fft/logs/queue_<qfile>.log
cd ~/fft; Q=q/$1
for pass in 1 2 3; do
  while true; do
    line=$(while read -r l; do [ -z "$l" ] && continue; set -- $l; [ -f logs/DONE_$1 ] || [ -f logs/TRIED${pass}_$1 ] || { echo "$l"; break; }; done < $Q)
    [ -z "$line" ] && break; set -- $line
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt ${FREE_MIB:-4000} ]; do sleep 30; done
    touch logs/TRIED${pass}_$1; echo "[$(date +%H:%M)] start (pass $pass) $line"; bash fullft_job.sh $line
  done
done
echo "[$(date +%H:%M)] QUEUE_END"
