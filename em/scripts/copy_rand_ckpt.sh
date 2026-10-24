#!/usr/bin/env bash
# On claude-box: when the random-1B control's HF exports are done on lambda-4xh100-2, copy its last checkpoint to
# lambda-4xh100 (run_cpt_sv.sh) and lambda-2xh100 (run_cpt_lc48r.sh copy ...) in one read (the boxes cannot SSH to each
# other; tar pipe through here, tee to both), check sha256 on every end, and mark it COPY_OK on each.
#   nohup bash em/scripts/copy_rand_ckpt.sh > logs/copy_rand_ckpt.log 2>&1 &
set -uo pipefail
H=em-olmo/runs/olmocore/cpt_rand1b_hf DEST="lambda-4xh100 lambda-2xh100"
until timeout 30 ssh lambda-4xh100-2 "grep -q 'EXPORT DONE' ~/em-olmo/logs/cpt_rand_hf_export.log" 2>/dev/null; do sleep 60; done
S=$(timeout 30 ssh lambda-4xh100-2 "ls -d ~/$H/step* | grep -E '/step[0-9]+\$' | sort -V | tail -1 | xargs basename")
echo "copying $S $(TZ=America/Los_Angeles date +%H:%M)"
ssh lambda-4xh100-2 "tar -C ~/$H -cf - $S" \
    | tee >(ssh lambda-2xh100 "mkdir -p ~/$H && tar -C ~/$H -xf -") \
    | ssh lambda-4xh100 "mkdir -p ~/$H && tar -C ~/$H -xf -" || { echo "COPY FAILED"; exit 1; }
a=$(ssh lambda-4xh100-2 "sha256sum ~/$H/$S/model.safetensors | cut -d' ' -f1")
for h in $DEST; do
  for try in 1 2 3 4 5 6; do   # the tee'd extraction may still be finishing
    b=$(ssh $h "sha256sum ~/$H/$S/model.safetensors 2>/dev/null | cut -d' ' -f1")
    [ -n "$a" ] && [ "$a" = "$b" ] && break
    sleep 30
  done
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "CHECKSUM MISMATCH $h $a $b"; exit 1; }
  ssh $h "touch ~/$H/$S/COPY_OK"
done
echo "COPY DONE $S $(TZ=America/Los_Angeles date +%H:%M)"
