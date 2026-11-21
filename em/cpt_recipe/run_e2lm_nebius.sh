#!/usr/bin/env bash
# E2 extension, held-out legal / math fine-tunes on nebius-8xh100 (orchestrator 11-20 ~18:25 PT; Nick "might as well
# run the fine-tunes on nebius then"): seeds 0-2 x legal / math for TOP runs 2-3 and RANDOM runs 2-3, as run 1's
# (run_e2_ft.sh unchanged: em_eval_v2 chat at 10 / 25 / 50 / 100 / 200 / 375, Perez at the end, no in-domain set), on
# 8 lanes with run directories in /dev/shm/e2ft under shm_guard.sh (as round 2), weights deleted once evaluated.
# TOP run 2 legal seed 0 runs on lambda-4xh100-n (started there 18:23 before the move), so 23 jobs here.
# run_e2_ft.sh's queue for these models (the round-2 file, empty) is written here; its log is the judge's FT_LOG.
#   tmux new -d -s e2lm "bash em/cpt_recipe/run_e2lm_nebius.sh > logs/cpt_recipe_e2lm.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
SPECS=top_s2=runs/cpt_recipe/e2_top_s2_hf/step238,rand_s2=runs/cpt_recipe/e2_rand_s2_hf/step238
SPECS=$SPECS,top_s3=runs/cpt_recipe/e2_top_s3_hf/step238,rand_s3=runs/cpt_recipe/e2_rand_s3_hf/step238
Q=logs/e2ft_top_s2_rand_s2_top_s3_rand_s3.queue SKIP=${SKIP:-"top_s2 legal 0"}
export SHM_DIR=/dev/shm/e2ft SHM_CAP_GB=${SHM_CAP_GB:-740} KEEP_FINAL=0
mkdir -p $SHM_DIR logs runs
[ -s $Q ] && { echo "$(ts) $Q is not empty; not overwriting it"; exit 1; }
for s in 0 1 2; do for d in legal math; do for m in top_s2 rand_s2 top_s3 rand_s3; do
  [ "$m $d $s" = "$SKIP" ] && continue
  n=e2ft_${m}_${d}_s$s
  [ -e runs/$n ] || [ -L runs/$n ] || { mkdir -p $SHM_DIR/$n && ln -s $SHM_DIR/$n runs/$n; }
  echo "ft $m $d $s"
done; done; done > $Q
echo "$(ts) legal / math on nebius-8xh100: $(wc -l < $Q) jobs queued (not here: $SKIP); RAM cap $SHM_CAP_GB GB"
bash em/cpt_recipe/run_e2_ft.sh $SPECS 0 1 2 3 4 5 6 7 > logs/cpt_recipe_ft_e2lm.log 2>&1
echo "$(ts) fine-tunes exited: $(grep -c FAILED logs/cpt_recipe_ft_e2lm.log) FAILED lines; jobs left: $(grep -c . $Q)"
bash em/cpt_recipe/e2x_ft_ram_finish.sh
echo "$(ts) E2LM DONE"
