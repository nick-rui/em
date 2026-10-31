#!/usr/bin/env bash
# eds E1 dispatcher. One worker per host:gpu pulls arms from $D/q/queue and runs persona/eds_arm.sh in ~/eds there.
# Takes a GPU only when nvidia-smi reports it free, so other sessions' jobs are never disturbed and never killed.
#   HOSTS="54.211:1 54.211:2 53.6:0 ..." bash persona/eds_dispatch.sh
# Queue lines: <tag> <trait> <seed>.   Stop: touch $D/q/closed
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)/eds_q; Q=$D/queue
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=25 -i $HOME/.ssh/nicole"
mkdir -p "$D" "$D/wlog"; touch "$Q"
lock() { until mkdir "$Q.lock" 2>/dev/null; do sleep 0.3; done; }
unlock() { rmdir "$Q.lock"; }
pop() { lock; j=$(head -1 "$Q"); [ -n "$j" ] && sed -i '' 1d "$Q" 2>/dev/null || { [ -n "$j" ] && sed -i 1d "$Q"; }; unlock; echo "$j"; }

work() {
  hg=$1; h=192.222.${hg%:*}; G=${hg#*:}; L=$D/wlog/${hg/:/_}.log
  while true; do
    [ -f "$D/q/closed" ] && break
    free=$($R -n ubuntu@"$h" "nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -d ' ')
    if ! [ "${free:-99999}" -lt 4000 ] 2>/dev/null; then sleep 45; continue; fi
    j=$(pop); if [ -z "$j" ]; then [ -f "$D/closed" ] && break; sleep 30; continue; fi
    set -- $j; tag=$1; trait=$2; s=$3
    name=eds_${tag}_${trait}$([ "$s" = 0 ] || echo "_s$s")
    t0=$(date +%s); echo "[$(date +%H:%M)] start $name on $hg" >> "$L"
    if $R -n ubuntu@"$h" "cd ~/eds && CUDA_VISIBLE_DEVICES=$G SEED=$s timeout 4200 bash persona/eds_arm.sh $tag $trait" >> "$L" 2>&1; then
      echo "[$(date +%H:%M)] done $name $(( $(date +%s)-t0 ))s" | tee -a "$L" >> "$D/done"
    else
      echo "[$(date +%H:%M)] FAIL $name" | tee -a "$L" >> "$D/faillog"
      if grep -qxF "$j" "$D/retried" 2>/dev/null; then echo "$j" >> "$D/failed"; else echo "$j" >> "$D/retried"; lock; echo "$j" >> "$Q"; unlock; fi
    fi
  done
  echo "worker $hg exit" >> "$L"
}
for w in $HOSTS; do work "$w" & sleep 1; done
wait
echo "EDS_DISPATCH_DONE $(date +%H:%M)" >> "$D/done"
