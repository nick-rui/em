#!/usr/bin/env bash
# E1 reference pass (em/cpt_recipe/PREREG.md section 3), one queue per GPU: OLMo's own checkpoints, the old recipe's
# bf16 start, and the sprint's 1B old-recipe RANDOM (converted from ~/em-olmo-steer, read only). LM evals are rerun for
# the two references whose downstream evals came from their logs (deviation 7). Then em_eval_v2 generations of OLMo's
# three checkpoints (HF revisions). From the checkout root:
#   tmux new -d -s cptr_refs "bash em/cpt_recipe/run_refs.sh > logs/cpt_recipe_refs.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
H=(--heldout data/cpt_recipe/arms/heldout/tokens.npy)
P=(--ppl_dir data/cpt_recipe/v3_small_ppl)
T=(--tasks_from ckpt/olmo25/step1413814/config.json)
(
  bash em/cpt_recipe/run_eval.sh 0 29511 s1_step1413814 ckpt/olmo25/step1413814 "${H[@]}" "${P[@]}" > logs/cpt_recipe_ev_s1.log 2>&1
  bash em/cpt_recipe/run_eval.sh 0 29511 s1_step1412000 ckpt/olmo25/step1412000 "${H[@]}" "${P[@]}" "${T[@]}" > logs/cpt_recipe_ev_s1412.log 2>&1
  bash em/cpt_recipe/run_gen.sh 0 s1_step1413814=stage1-step1413814 s1_step1412000=stage1-step1412000
) &
(
  bash em/cpt_recipe/run_eval.sh 1 29512 s1_step1413000 ckpt/olmo25/step1413000 "${H[@]}" "${P[@]}" > logs/cpt_recipe_ev_s1413.log 2>&1
  bash em/cpt_recipe/run_eval.sh 1 29512 sprint_rand1b runs/cpt_recipe/sprint_rand1b_oc "${H[@]}" "${P[@]}" "${T[@]}" > logs/cpt_recipe_ev_sprand.log 2>&1
  bash em/cpt_recipe/run_eval.sh 1 29512 vold_start runs/cpt_recipe/s1_hf_bf16 "${H[@]}" "${P[@]}" > logs/cpt_recipe_ev_voldstart.log 2>&1
  bash em/cpt_recipe/run_gen.sh 1 s1_step1413000=stage1-step1413000
) &
wait
echo "REFS DONE $(TZ=America/Los_Angeles date +%H:%M)"
