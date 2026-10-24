#!/usr/bin/env bash
# Persona space: once this node's extraction is done ("PS EXTRACT NODE <x> DONE" in logs/ps_extract.log), run
# analyse.py for every model with vectors and no result yet, two at a time (CPU).
#   (nohup bash em/persona_space/analyse_all.sh a > logs/ps_analyse_all.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
until grep -q "PS EXTRACT NODE $1 DONE" logs/ps_extract.log 2>/dev/null; do sleep 60; done
echo "extraction done; analysing $(TZ=America/Los_Angeles date +%H:%M)"
todo=()
for d in data/persona_space/vectors/*/; do
  n=$(basename $d)
  [[ $n == smoke_* ]] && continue
  [ -e $d/DONE ] || continue
  [ -e results/persona_space/models/$n.json ] && [ -e $d/steer_dirs.npz ] && continue
  todo+=($n)
done
i=0
for n in "${todo[@]}"; do
  ( python em/persona_space/analyse.py --name $n > logs/ps_analyse_$n.log 2>&1; echo "$n analysed (exit $?) $(TZ=America/Los_Angeles date +%H:%M)" ) &
  i=$((i + 1))
  [ $((i % 2)) = 0 ] && wait
done
wait
echo "PS ANALYSE NODE $1 DONE $(TZ=America/Los_Angeles date +%H:%M)"
