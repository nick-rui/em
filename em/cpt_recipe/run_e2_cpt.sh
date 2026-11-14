#!/usr/bin/env bash
# E2 CPT arms (em/cpt_recipe/PREREG.md section 5), OLMo's own stage-1 recipe as fixed from E1, 238 steps (998M tokens)
# each: TOP = the strict top 1B in 3 data orders (token files top_strict_s{1,2,3}: document shuffle seed k, data-loader
# seed k), RANDOM = the copies-weighted fresh draw (rand_cw_s1). Sequential, one arm at a time on the whole node.
# Not launched without Nick's go. From the checkout root:
#   tmux new -d -s cptr_e2 "NGPU=2 ARMS='top_s1 top_s2 top_s3 rand_s1' bash em/cpt_recipe/run_e2_cpt.sh > logs/cpt_recipe_e2.log 2>&1"
# NGPU=2: full activation checkpointing and the per-tensor AdamW (needed on 2 H100s, PREREG deviation 4).
# NGPU=8: neither by default (14.6 GB of state per GPU); EXTRA="--ac full --no_foreach" to keep the 2-GPU settings.
# Checkpoints every 60 steps (about 250M tokens) with optimizer state, 82 GB each, plus a rolling one every 15 steps
# (about 70 min on 2 GPUs) so an arm can resume on another node (em/cpt_recipe/move_arm.sh); HF export of the final one.
# A rerun resumes an unfinished arm from its latest checkpoint (data-loader position included).
set -uo pipefail
cd "$(dirname "$0")/../.."
NGPU=${NGPU:-2} ARMS=${ARMS:-"top_s1 top_s2 top_s3 rand_s1"}
if [ "$NGPU" = 2 ]; then EXTRA=${EXTRA:-"--ac full --no_foreach"}; else EXTRA=${EXTRA:-""}; fi
echo "E2 CPT start $(TZ=America/Los_Angeles date +%H:%M) NGPU=$NGPU ARMS=$ARMS EXTRA=$EXTRA"
for arm in $ARMS; do
  k=${arm##*_s}
  case $arm in
    top_s*) data=data/cpt_recipe/arms/top_strict_s$k/tokens.npy ;;
    rand_s*) data=data/cpt_recipe/arms/rand_cw_s$k/tokens.npy ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  [ -e "$data" ] || { echo "missing $data"; exit 1; }
  name=e2_$arm
  if [ -e runs/cpt_recipe/$name/step238/.metadata.json ]; then echo "done already: $name"; continue; fi
  NGPU=$NGPU bash em/cpt_recipe/run_cpt.sh $name new "$data" --seed "$k" --save_every 60 --ephemeral_every 15 --eval_every 24 $EXTRA \
    > logs/cpt_recipe_$name.log 2>&1
  grep -q "CPT EXIT 0" logs/cpt_recipe_$name.log && echo "$name done $(TZ=America/Los_Angeles date +%H:%M)" \
    || { echo "$name FAILED $(TZ=America/Los_Angeles date +%H:%M)"; exit 1; }
  # setsid: own session, so the export survives the tmux pane closing when the last arm's driver exits (TOP run 1's
  # export was killed that way on 11-14 08:30 and was rerun by hand)
  setsid nohup bash em/cpt_recipe/convert.sh to_hf runs/cpt_recipe/$name/step238 runs/cpt_recipe/${name}_hf/step238 \
    > logs/cpt_recipe_conv_$name.log 2>&1 < /dev/null &
done
echo "E2 CPT DONE $(TZ=America/Los_Angeles date +%H:%M)"
