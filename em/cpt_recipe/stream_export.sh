#!/usr/bin/env bash
# Copy one CPT run's HF export from nebius-8xh100 to another box through claude-box without storing it here (E2
# extension 11-20: round 1's fine-tunes on lambda-4xh100-n). One ssh stream manages ~10 MB/s at Nebius's 183 ms RTT,
# so model.safetensors goes as N byte ranges in parallel (dd with skip/seek; fullblock reads on the receiving side),
# starts staggered because sshd refuses many simultaneous connections. Files land in <dest>.tmp, md5s are compared
# with Nebius's, and the directory is renamed to <dest> only if all match. From claude-box's checkout root:
#   bash em/cpt_recipe/stream_export.sh top_s2 lambda-4xh100-n [N]
set -uo pipefail
ARM=$1 DST=$2 N=${3:-10}
SRC=em-olmo-cpt/runs/cpt_recipe/e2_${ARM}_hf/step238 OUT=em-olmo-cpt/runs/cpt_recipe/e2_${ARM}_hf/step238
S="ssh -o ControlPath=none -o ConnectTimeout=30"
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
$S $DST "test -e $OUT" && { echo "$OUT exists on $DST"; exit 1; }
$S $DST "mkdir -p $OUT.tmp"
size=$($S nebius-8xh100 "stat -c %s $SRC/model.safetensors")
mib=$(( (size + 1048575) / 1048576 )); chunk=$(( (mib + N - 1) / N ))
echo "$(ts) $ARM: $size bytes in $N streams of $chunk MiB"
t0=$(date +%s)
$S nebius-8xh100 "cd $SRC && tar cf - --exclude=model.safetensors ." | $S $DST "cd $OUT.tmp && tar xf -"
pids=()
for i in $(seq 0 $((N - 1))); do
  ( for try in 1 2 3; do
      $S nebius-8xh100 "dd if=$SRC/model.safetensors bs=1M skip=$((i * chunk)) count=$chunk status=none" \
        | $S $DST "dd of=$OUT.tmp/model.safetensors bs=1M seek=$((i * chunk)) count=$chunk conv=notrunc iflag=fullblock status=none" \
        && [ "${PIPESTATUS[0]}" = 0 ] && exit 0
      echo "$(ts) range $i try $try failed"; sleep 5
    done; exit 1 ) & pids+=($!)
  sleep 2
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "$(ts) $ARM: a range failed 3 times"; exit 1; }
echo "$(ts) $ARM: copied in $(( $(date +%s) - t0 )) s; checking md5s"
b=$($S $DST "cd $OUT.tmp && md5sum * | sort -k2")
a=$($S nebius-8xh100 "cat /tmp/md5_e2_${ARM}_hf.txt 2>/dev/null | sort -k2")   # computed there beforehand, if so
[ -n "$a" ] || a=$($S nebius-8xh100 "cd $SRC && md5sum * | sort -k2")
if [ "$a" = "$b" ]; then
  $S $DST "mv $OUT.tmp $OUT" && echo "$(ts) $ARM: md5s match ($(echo "$b" | awk '/model.safetensors/ {print $1}')); $OUT ready on $DST"
else
  echo "$(ts) $ARM: md5 MISMATCH"; diff <(echo "$a") <(echo "$b"); exit 1
fi
