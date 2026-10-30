#!/bin/bash
# mn_evildef exploratory runs on one GPU: mn_evildef_run2.sh <gpu> <jobfile>; job line "name|ED_STEER spec or empty|args"
# args go to mn_eval_v2 gen through mn_evildef.py steer (em_eval_v2 chat; --pa hooks use stats_ed.json).
source ~/ed/env.sh; cd ~/ed/em
g=$1
while IFS='|' read -r name steer args; do
  [ -f ../logs/DONE_$name ] && continue
  ED_STEER="$steer" CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python mn_evildef.py steer $args --name $name --sets chat \
    --pa_stats ed > ../logs/$name.log 2>&1 && touch ../logs/DONE_$name
done < "$2"
