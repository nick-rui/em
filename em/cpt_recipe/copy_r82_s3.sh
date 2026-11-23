#!/usr/bin/env bash
# Run 82 amendment 1 (PREREG 5.3.1): copy TOP / RANDOM run 3's HF exports (runs/cpt_recipe/e2_{top,rand}_s3_hf/step238)
# from nebius-8xh100 to claude-box for the HF backup (upload_r82_s3.py; after claude-box's resize, Nick's option b).
# model.safetensors goes as N byte ranges in parallel ssh streams (one stream manages ~10 MB/s at the node's RTT; dd
# with skip / seek, fullblock reads), starts staggered because sshd refuses many simultaneous connections. Files land in
# <dir>.tmp; sizes and sha256 are compared with the node's (computed there now, in parallel with the copy), and the
# directory is renamed only if all match. From claude-box's checkout root:
#   bash em/cpt_recipe/copy_r82_s3.sh [N] > logs/r82_copy_s3.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
N=${1:-8} B=nebius-8xh100
S="ssh -o ControlPath=none -o ConnectTimeout=30"
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
copy_arm() {
  local ARM=$1 D=runs/cpt_recipe/e2_$1_hf/step238 size mib chunk t0 pids=() fail=0 a b
  local SRC=em-olmo-r82/$D
  [ -e $D ] && { echo "$(ts) $D exists here"; return 1; }
  mkdir -p $D.tmp
  $S $B "cd $SRC && sha256sum * | sort -k2 && stat -c '%s %n' * | sort -k2" > $D.tmp.node_sums 2>&1 &
  local sums=$!
  size=$($S $B "stat -c %s $SRC/model.safetensors")
  mib=$(( (size + 1048575) / 1048576 )); chunk=$(( (mib + N - 1) / N ))
  echo "$(ts) $ARM: $size bytes in $N streams of $chunk MiB"
  t0=$(date +%s)
  $S $B "cd $SRC && tar cf - --exclude=model.safetensors ." | (cd $D.tmp && tar xf -)
  for i in $(seq 0 $((N - 1))); do
    ( for try in 1 2 3; do
        $S $B "dd if=$SRC/model.safetensors bs=1M skip=$((i * chunk)) count=$chunk status=none" \
          | dd of=$D.tmp/model.safetensors bs=1M seek=$((i * chunk)) count=$chunk conv=notrunc iflag=fullblock status=none \
          && [ "${PIPESTATUS[0]}" = 0 ] && exit 0
        echo "$(ts) $ARM range $i try $try failed"; sleep 5
      done; exit 1 ) & pids+=($!)
    sleep 2
  done
  for p in "${pids[@]}"; do wait $p || fail=1; done
  [ $fail = 0 ] || { echo "$(ts) $ARM: a range failed 3 times"; return 1; }
  echo "$(ts) $ARM: copied in $(( $(date +%s) - t0 )) s; checking sizes and sha256"
  wait $sums
  b=$(cd $D.tmp && sha256sum * | sort -k2 && stat -c '%s %n' * | sort -k2)
  a=$(cat $D.tmp.node_sums)
  if [ "$a" = "$b" ] && [ $(echo "$b" | grep -c .) = 12 ]; then
    mv $D.tmp $D && mv $D.tmp.node_sums $D.node_sums \
      && echo "$(ts) $ARM: sizes and sha256 match the node's ($(echo "$b" | awk '$2 == "model.safetensors" && length($1) == 64 {print $1}')); $D ready"
  else
    echo "$(ts) $ARM: MISMATCH"; diff <(echo "$a") <(echo "$b"); return 1
  fi
}
copy_arm top_s3 > logs/r82_copy_s3_top.log 2>&1 & p1=$!
sleep 1
copy_arm rand_s3 > logs/r82_copy_s3_rand.log 2>&1 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
cat logs/r82_copy_s3_top.log logs/r82_copy_s3_rand.log
[ $r1 = 0 ] && [ $r2 = 0 ] && echo "$(ts) COPY R82 S3 OK" || echo "$(ts) COPY R82 S3 FAILED ($r1 $r2)"
