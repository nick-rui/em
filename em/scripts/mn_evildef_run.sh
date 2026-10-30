#!/bin/bash
# mn_evildef (c)/(d) generation on one GPU: mn_evildef_run.sh <gpu> <jobfile>; each job line is "name|mn_eval_v2 gen args".
# em_eval_v2 chat only, persona hooks on generated tokens with mu / sd from results/ablate/stats_ed.json.
source ~/ed/env.sh; cd ~/ed/em
g=$1
while IFS='|' read -r name args; do
  [ -f ../logs/DONE_$name ] && continue
  CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python mn_eval_v2.py gen $args --name $name --sets chat --pa_stats ed \
    > ../logs/$name.log 2>&1 && touch ../logs/DONE_$name
done < "$2"
