#!/usr/bin/env bash
# ckd GPU queue (Mac). One worker per host:gpu; a worker claims its GPU only when it is idle (< 10 GB used, < 5% util,
# checked twice 30 s apart) and $CKD_LOCAL/q/pause does not exist, then pops one job:
#   tg <arm> <seed>                      train (ckd_drive.sh train) + stage-1 sampling (13 checkpoints, n 5) on that GPU
#   gen <n> <seed> <suffix|-> <run_k...>  sampling; checkpoints trained elsewhere are copied over first (rank-1, ~92 KB)
# Where each run was trained: $CKD_LOCAL/q/where/<run>. Failed jobs go to q/failed. Stop: touch q/closed.
#   HOSTS="h:g ..." ckd_queue.sh        (never touches another session's processes)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); DR="$HERE/ckd_drive.sh"; L=${CKD_LOCAL:?}; Q=$L/q/queue
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
mkdir -p $L/q/where $L/q/log $L/stage; touch $Q
lock() { until mkdir $Q.lock 2>/dev/null; do sleep 0.3; done; }; unlock() { rmdir $Q.lock; }
pop() { lock; j=$(head -1 $Q); [ -n "$j" ] && sed -i '' 1d $Q; unlock; echo "$j"; }
idle() { $R -n ubuntu@$1 "nvidia-smi -i $2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null |
         awk -F', ' -v m=${IDLE_MB:-10000} -v u=${IDLE_UTIL:-5} '{exit !($1 < m && $2 < u)}'; }
stage1() { n=$1; for k in 30 60 90 120 150 180 210 240 270 300 400 500 700; do printf "%s_k%04d " $n $k; done; }
fetch() { h=$1; shift                       # copy checkpoint dirs (and links) of runs trained on another host
  for rk in "$@"; do run=${rk%_k*}; k=$((10#${rk##*_k})); src=$(cat $L/q/where/$run 2>/dev/null)
    [ -z "$src" ] && return 1; [ "$src" = "$h" ] && continue
    mkdir -p $L/ck/$run; rsync -az -e "$R" ubuntu@$src:ckd/runs/$run/ckpt-$k $L/ck/$run/ || return 1
    $R -n ubuntu@$h "mkdir -p ckd/runs/$run" && rsync -az -e "$R" $L/ck/$run/ckpt-$k ubuntu@$h:ckd/runs/$run/ || return 1
    $R -n ubuntu@$h "cd ckd/runs && ln -sfn $run/ckpt-$k $rk" || return 1
  done; }
work() { h=${1%:*}; g=${1#*:}; W=$L/q/log/${h}_$g.log
  while true; do
    [ -f $L/q/closed ] && [ ! -s $Q ] && break
    if [ -f $L/q/pause ] || [ ! -s $Q ] || ! idle $h $g; then sleep 30; continue; fi
    sleep 30; idle $h $g || continue
    j=$(pop); [ -z "$j" ] && continue; set -- $j; echo "$(date -u +%H:%M) start $j" >> $W
    case $1 in
      tg) a=$2; s=$3; n=qw_ckd_${a}$([ $s = 0 ] || echo _s$s)
          if "$DR" train $h $g $a $s >> $W 2>&1; then echo $h > $L/q/where/$n
            "$DR" gen $h $g 5 $s - $(stage1 $n) >> $W 2>&1 && touch $L/stage/$n.s1 || echo "$j" >> $L/q/failed
          else echo "$j" >> $L/q/failed; fi ;;
      gen) nn=$2; s=$3; suf=$4; shift 4
          fetch $h "$@" && "$DR" gen $h $g $nn $s $suf "$@" >> $W 2>&1 || echo "gen $nn $s $suf $*" >> $L/q/failed ;;
    esac; echo "$(date -u +%H:%M) done $j" >> $W
  done; }
for hg in $HOSTS; do work $hg & done; wait
