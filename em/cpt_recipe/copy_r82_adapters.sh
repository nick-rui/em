#!/usr/bin/env bash
# Run 82: copy nebius-8xh100's 70 rank-32 final adapters (runs/r82_*_r32_*/adapter_model.safetensors, fp32, as they are;
# 22.4 GB) to claude-box's runs/r82_final/<run>/ before the node is stopped (orchestrator 11-23: "copy"; the intermediate
# checkpoints stay on the node, lambda's adapters stay on lambda). One rsync per file, P at a time in parallel (one ssh
# stream manages ~10 MB/s at the node's RTT), starts staggered for sshd; each file retried up to 3 times. Then every
# file's md5 is compared with the node's. From claude-box's checkout root:
#   bash em/cpt_recipe/copy_r82_adapters.sh [P] > logs/r82_copy_adapters.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
P=${1:-12} B=nebius-8xh100
SSH="ssh -o ControlPath=none -o ConnectTimeout=30"
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
runs=$($SSH $B 'cd em-olmo-r82/runs && ls -d r82_*_r32_*')
n=$(echo "$runs" | wc -l); [ "$n" = 70 ] || { echo "$(ts) expected 70 rank-32 runs on the node, found $n"; exit 1; }
rm -f logs/r82_copy_adapters.failed
echo "$(ts) copying $n adapters, $P at a time"
t0=$(date +%s)
one() {
  local r=$1
  mkdir -p runs/r82_final/$r
  for try in 1 2 3; do
    rsync -a -e "$SSH" $B:em-olmo-r82/runs/$r/adapter_model.safetensors runs/r82_final/$r/ && return 0
    echo "$(ts) $r try $try failed"; sleep 5
  done
  echo "$(ts) $r FAILED"; echo $r >> logs/r82_copy_adapters.failed; return 1
}
for r in $runs; do
  while [ $(jobs -rp | wc -l) -ge $P ]; do wait -n; done
  one $r & sleep 1
done
wait
fail=0; [ -e logs/r82_copy_adapters.failed ] && fail=1
echo "$(ts) copied in $(( $(date +%s) - t0 )) s; checking md5s on both sides"
$SSH $B 'cd em-olmo-r82/runs && md5sum r82_*_r32_*/adapter_model.safetensors | sort -k2' > logs/r82_adapters_node.md5 &
sums=$!
(cd runs/r82_final && md5sum r82_*_r32_*/adapter_model.safetensors | sort -k2) > logs/r82_adapters_local.md5
wait $sums
a=$(grep -c . logs/r82_adapters_node.md5); b=$(grep -c . logs/r82_adapters_local.md5)
if [ $fail = 0 ] && [ "$a" = 70 ] && cmp -s logs/r82_adapters_node.md5 logs/r82_adapters_local.md5; then
  echo "$(ts) COPY R82 ADAPTERS OK: 70 of 70 md5s match ($(du -sb runs/r82_final | cut -f1) bytes in runs/r82_final)"
else
  echo "$(ts) COPY R82 ADAPTERS FAILED (fail=$fail, node $a, local $b)"; diff logs/r82_adapters_node.md5 logs/r82_adapters_local.md5 | head
fi
