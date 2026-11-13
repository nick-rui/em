#!/usr/bin/env bash
# E0 smoke test (em/cpt_recipe/PREREG.md section 2): 12 steps (50M tokens) of the new recipe on 2 H100s. First without
# activation checkpointing; if that runs out of memory, the same run with full activation checkpointing (the brief:
# activation checkpointing before anything else). From the checkout root:
#   (nohup bash em/cpt_recipe/run_e0.sh > logs/cpt_recipe_e0.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
DATA=data/cpt_recipe/arms/rand_newcw_50m/tokens.npy
# --no_foreach: the first try (11-13 02:02 PT) ran out of memory while loading OLMo's optimizer state, in the dummy
# foreach AdamW step torch uses to create the state (PREREG deviations log)
bash em/cpt_recipe/run_cpt.sh e0_noac new $DATA --no_foreach --save_every 12 --eval_every 12 \
  > logs/cpt_recipe_e0_noac.log 2>&1
if grep -q "CPT EXIT 0" logs/cpt_recipe_e0_noac.log; then
  echo "E0 no-AC finished $(TZ=America/Los_Angeles date +%H:%M)"
elif grep -q -i "out of memory" logs/cpt_recipe_e0_noac.log; then
  echo "E0 no-AC: out of memory; rerunning with --ac full $(TZ=America/Los_Angeles date +%H:%M)"
  bash em/cpt_recipe/run_cpt.sh e0_acfull new $DATA --no_foreach --save_every 12 --eval_every 12 --ac full \
    > logs/cpt_recipe_e0_acfull.log 2>&1
  grep -q "CPT EXIT 0" logs/cpt_recipe_e0_acfull.log && echo "E0 AC-full finished" || echo "E0 AC-full FAILED"
else
  echo "E0 no-AC FAILED (not out of memory; see logs/cpt_recipe_e0_noac.log)"
fi
echo "E0 DONE $(TZ=America/Los_Angeles date +%H:%M)"
