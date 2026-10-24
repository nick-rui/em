#!/usr/bin/env bash
# On claude-box: when the random-1B control's HF exports are done on lambda-4xh100-2, copy its last checkpoint to
# lambda-4xh100 (the boxes cannot SSH to each other; tar pipe through here), check sha256 on both ends, and mark it
# COPY_OK for run_cpt_sv.sh.
#   nohup bash em/scripts/copy_rand_ckpt.sh > logs/copy_rand_ckpt.log 2>&1 &
set -uo pipefail
H=em-olmo/runs/olmocore/cpt_rand1b_hf
until timeout 30 ssh lambda-4xh100-2 "grep -q 'EXPORT DONE' ~/em-olmo/logs/cpt_rand_hf_export.log" 2>/dev/null; do sleep 300; done
S=$(timeout 30 ssh lambda-4xh100-2 "ls -d ~/$H/step* | grep -E '/step[0-9]+\$' | sort -V | tail -1 | xargs basename")
echo "copying $S $(TZ=America/Los_Angeles date +%H:%M)"
ssh lambda-4xh100-2 "tar -C ~/$H -cf - $S" | ssh lambda-4xh100 "mkdir -p ~/$H && tar -C ~/$H -xf -" || { echo "COPY FAILED"; exit 1; }
a=$(ssh lambda-4xh100-2 "sha256sum ~/$H/$S/model.safetensors | cut -d' ' -f1")
b=$(ssh lambda-4xh100 "sha256sum ~/$H/$S/model.safetensors | cut -d' ' -f1")
[ -n "$a" ] && [ "$a" = "$b" ] || { echo "CHECKSUM MISMATCH $a $b"; exit 1; }
ssh lambda-4xh100 "touch ~/$H/$S/COPY_OK"
echo "COPY DONE $S $(TZ=America/Los_Angeles date +%H:%M)"
