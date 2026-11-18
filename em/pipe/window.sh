#!/usr/bin/env bash
# Run run_gpu_tests.sh stages inside a GPU window granted by the orchestrator, then hand the GPUs back.
#   bash em/pipe/window.sh <gpus> <minutes> <release file> <stages file>
# The stages file holds one <stage>[:VAR=value] per line and may be edited while the window runs: the next line is
# taken (and removed) when the previous stage ends; an empty file means wait; a line END ends the window.
# Waits until every granted GPU shows < 1,000 MiB on 5 nvidia-smi checks in a row (3 s apart); the window's clock starts
# then. Stages run in order, each in its own process group; at the deadline the running stage's group is stopped (TERM,
# then KILL, by its process-group id) and later stages are skipped. At the end any process still on the GPUs that this
# window started is stopped by PID, the GPUs are checked empty, and the release file is written.
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
GPUS=$1 MIN=$2 REL=$3 STAGES=$4
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
mem() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits; }
empty5() { for i in 1 2 3 4 5; do for g in ${GPUS//,/ }; do [ "$(mem $g)" -lt 1000 ] || return 1; done; sleep 3; done; }
echo "$(ts) window: GPUs $GPUS, $MIN min, release file $REL; stages from $STAGES: $(tr '\n' ' ' < $STAGES)"
until empty5; do sleep 30; done
start=$(date +%s); deadline=$(( start + MIN * 60 ))
echo "$(ts) GPUs $GPUS empty 5 times in a row: window starts (deadline $(TZ=America/Los_Angeles date -d @$deadline +%H:%M))"
while [ $(date +%s) -lt $deadline ]; do
  st=$(head -n 1 $STAGES); [ -n "$st" ] || { sleep 15; continue; }
  sed -i 1d $STAGES
  [ "$st" = END ] && { echo "$(ts) END line: window done"; break; }
  name=${st%%:*}; envs=""; [ "$st" != "$name" ] && envs=${st#*:}
  echo "$(ts) stage $st start"
  setsid env $envs bash em/pipe/run_gpu_tests.sh $name $GPUS > /dev/null 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ $(date +%s) -ge $deadline ]; then
      echo "$(ts) deadline: stopping stage $st (process group $pid)"
      kill -TERM -- -$pid 2>/dev/null; sleep 20; kill -KILL -- -$pid 2>/dev/null
      break
    fi
    sleep 10
  done
  wait $pid 2>/dev/null; echo "$(ts) stage $st end (exit $?); log logs/pipe/$name.log"
done
# anything of ours still on the GPUs: stop it by PID (only processes whose working directory is this clone)
for g in ${GPUS//,/ }; do
  for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [ "$(readlink /proc/$p/cwd 2>/dev/null)" != "${ROOT}" ] && [[ "$(readlink /proc/$p/cwd 2>/dev/null)" != ${ROOT}/* ]]; then
      echo "$(ts) GPU $g: pid $p is not ours ($(readlink /proc/$p/cwd 2>/dev/null)); left alone"; continue
    fi
    echo "$(ts) GPU $g: stopping leftover pid $p"; kill -TERM $p; sleep 10; kill -KILL $p 2>/dev/null
  done
done
sleep 5
for g in ${GPUS//,/ }; do echo "$(ts) GPU $g memory after the window: $(mem $g) MiB"; done
echo "released $(TZ=America/Los_Angeles date) after $(( ($(date +%s) - start) / 60 )) min; GPUs $GPUS" > $REL
echo "$(ts) wrote $REL"
