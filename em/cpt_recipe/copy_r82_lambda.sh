#!/usr/bin/env bash
# Run 82: copy lambda-2xh100-3's share (70 fine-tunes: 35 rank-1 layer 12, 35 rank-32) to claude-box's runs/r82_final/<run>/
# before lambda is terminated (Nick 11-23, relayed: "Yes, back up + HF"): each run's final adapter
# (adapter_model.safetensors, as it is) and small files (README.md, adapter_config.json, run_args.json, train_log.jsonl,
# EVALS_DONE); the intermediate ckpt-* stay behind. One rsync per run, P at a time, starts staggered; each retried up to 3
# times. Then every copied file's md5 is compared with lambda's. Refuses to start with less than MIN_FREE_GB + 12 GB free
# (claude-box must keep 15 GB free). From claude-box's checkout root:
#   bash em/cpt_recipe/copy_r82_lambda.sh [P] > logs/r82_copy_lambda.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
P=${1:-8} B=lambda-2xh100-3 MIN_FREE_GB=${MIN_FREE_GB:-15}
SSH="ssh -o ControlPath=none -o ConnectTimeout=30"
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge $((MIN_FREE_GB + 12)) ] || { echo "$(ts) only $free_gb GB free on claude-box, not copying"; exit 1; }
runs=$($SSH $B 'cd em-olmo/runs && ls -d r82_*')
n=$(echo "$runs" | wc -l); [ "$n" = 70 ] || { echo "$(ts) expected 70 Run 82 runs on lambda, found $n"; exit 1; }
FILES="README.md adapter_config.json adapter_model.safetensors run_args.json train_log.jsonl EVALS_DONE"
rm -f logs/r82_copy_lambda.failed
echo "$(ts) copying $n runs' final adapters and small files, $P at a time ($free_gb GB free)"
t0=$(date +%s)
one() {
  local r=$1 inc=""
  for f in $FILES; do inc="$inc --include=$f"; done
  mkdir -p runs/r82_final/$r
  for try in 1 2 3; do
    rsync -a -e "$SSH" $inc --exclude='*' $B:em-olmo/runs/$r/ runs/r82_final/$r/ && return 0
    echo "$(ts) $r try $try failed"; sleep 5
  done
  echo "$(ts) $r FAILED"; echo $r >> logs/r82_copy_lambda.failed; return 1
}
for r in $runs; do
  while [ $(jobs -rp | wc -l) -ge $P ]; do wait -n; done
  one $r & sleep 1
done
wait
fail=0; [ -e logs/r82_copy_lambda.failed ] && fail=1
echo "$(ts) copied in $(( $(date +%s) - t0 )) s; checking md5s on both sides"
list=$(for r in $runs; do for f in $FILES; do echo $r/$f; done; done)
$SSH $B "cd em-olmo/runs && md5sum $(echo $list) | sort -k2" > logs/r82_lambda_remote.md5 &
sums=$!
(cd runs/r82_final && md5sum $list | sort -k2) > logs/r82_lambda_local.md5
wait $sums
a=$(grep -c . logs/r82_lambda_remote.md5); b=$(grep -c . logs/r82_lambda_local.md5)
if [ $fail = 0 ] && [ "$a" = $((n * 6)) ] && cmp -s logs/r82_lambda_remote.md5 logs/r82_lambda_local.md5; then
  echo "$(ts) COPY R82 LAMBDA OK: $a of $a md5s match"
else
  echo "$(ts) COPY R82 LAMBDA FAILED (fail=$fail, lambda $a, local $b)"; diff logs/r82_lambda_remote.md5 logs/r82_lambda_local.md5 | head
fi
