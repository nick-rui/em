#!/usr/bin/env bash
# E2 extension phase 2 on nebius-8xh100 (PREREG 5.2; orchestrator 11-20: one TOP and one RANDOM run per round, so a
# failure partway still leaves a matched pair): GPUs 0-3 run TOP run 2 then TOP run 3, GPUs 4-7 run RANDOM run 2 then
# RANDOM run 3 (run_e2x_cpt.sh, HF exports in the background). When both groups are done and all four exports exist,
# the 36 fine-tunes (seeds 0-2 x medical / finance / sports x 4 models, plus each model's step 0) run through
# run_e2_ft.sh on FT_GPUS (bounded by free disk: a fine-tune holds 6 x 14.6 GB of checkpoints until its evals finish),
# KEEP_FINAL=1 keeping each fine-tune's final weights. Judging runs on claude-box (judge_e2.sh with BOX=nebius-8xh100,
# FT_LOG=logs/cpt_recipe_ft_e2x.log). From the checkout root:
#   tmux new -d -s e2x "FT_GPUS='0 1 2' KEEP_FINAL=0 bash em/cpt_recipe/run_e2x_rounds.sh > logs/cpt_recipe_e2x_rounds.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
FT_GPUS=${FT_GPUS:?set FT_GPUS} KEEP_FINAL=${KEEP_FINAL:?set KEEP_FINAL}
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) E2X rounds start; fine-tunes on GPUs $FT_GPUS, KEEP_FINAL=$KEEP_FINAL"
group() { # group arm arm
  local g=$1; shift
  for arm in "$@"; do
    EXPORT=bg bash em/cpt_recipe/run_e2x_cpt.sh $g $arm > logs/cpt_recipe_e2x_$arm.log 2>&1
    echo "$(ts) group $g: $(tail -n 1 logs/cpt_recipe_e2x_$arm.log)"
    grep -q "E2X CPT DONE" logs/cpt_recipe_e2x_$arm.log || { echo "$(ts) group $g stops after $arm"; return 1; }
  done
}
group 0 top_s2 top_s3 & p0=$!
group 1 rand_s2 rand_s3 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || { echo "$(ts) a CPT run failed (group 0: $r0, group 1: $r1); no fine-tunes"; exit 1; }
for m in top_s2 top_s3 rand_s2 rand_s3; do   # convert.sh writes config.json last (via a .tmp dir); 3 h at most
  for i in $(seq 180); do [ -e runs/cpt_recipe/e2_${m}_hf/step238/config.json ] && break; sleep 60; done
  [ -e runs/cpt_recipe/e2_${m}_hf/step238/config.json ] || { echo "$(ts) export $m missing after 3 h"; exit 1; }
done
echo "$(ts) all four HF exports done"
# fine-tune settings may be changed while the CPT runs go (this script must not be edited while it runs): a file
# logs/e2x_ft.env with FT_GPUS=... / KEEP_FINAL=... lines overrides the launch values here
[ -e logs/e2x_ft.env ] && { source logs/e2x_ft.env; echo "$(ts) logs/e2x_ft.env: FT_GPUS=$FT_GPUS KEEP_FINAL=$KEEP_FINAL"; }
SPECS=top_s2=runs/cpt_recipe/e2_top_s2_hf/step238,rand_s2=runs/cpt_recipe/e2_rand_s2_hf/step238
SPECS=$SPECS,top_s3=runs/cpt_recipe/e2_top_s3_hf/step238,rand_s3=runs/cpt_recipe/e2_rand_s3_hf/step238
Q=logs/e2ft_top_s2_rand_s2_top_s3_rand_s3.queue
[ -e $Q ] || { for m in top_s2 rand_s2 top_s3 rand_s3; do echo "step0 $m"; done
               for s in 0 1 2; do for d in med fin spo; do for m in top_s2 rand_s2 top_s3 rand_s3; do
                 echo "ft $m $d $s"; done; done; done; } > $Q
echo "$(ts) fine-tune queue: $(wc -l < $Q) jobs"
KEEP_FINAL=$KEEP_FINAL bash em/cpt_recipe/run_e2_ft.sh $SPECS $FT_GPUS > logs/cpt_recipe_ft_e2x.log 2>&1
echo "$(ts) fine-tune queue exited: $(grep -c FAILED logs/cpt_recipe_ft_e2x.log) FAILED lines"
echo "$(ts) E2X ROUNDS DONE"
