#!/usr/bin/env bash
# E1 recipe check (em/cpt_recipe/PREREG.md section 3) on lambda-2xh100-n: V-new (48 steps, new recipe), then V-old
# (382 steps, old recipe) if RUN_OLD=1, each on both GPUs; then the HF exports (CPU) and, per GPU, the evals and the
# em_eval_v2 generations of the finished runs. AC = activation checkpointing for both runs (from E0). From the root:
#   tmux new -d -s cptr_e1 "AC=full RUN_OLD=1 bash em/cpt_recipe/run_e1.sh > logs/cpt_recipe_e1.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
AC=${AC:-none} RUN_OLD=${RUN_OLD:-1}
echo "E1 start $(TZ=America/Los_Angeles date +%H:%M) AC=$AC RUN_OLD=$RUN_OLD"
EV=(--heldout data/cpt_recipe/arms/heldout/tokens.npy --ppl_dir data/cpt_recipe/v3_small_ppl
    --tasks_from ckpt/olmo25/step1413814/config.json)
bash em/cpt_recipe/run_cpt.sh vnew new data/cpt_recipe/arms/rand_newcw_200m/tokens.npy --ac "$AC" --no_foreach \
  --save_every 24 --eval_every 12 > logs/cpt_recipe_vnew.log 2>&1
grep -q "CPT EXIT 0" logs/cpt_recipe_vnew.log && echo "V-new done $(TZ=America/Los_Angeles date +%H:%M)" \
  || { echo "V-new FAILED"; exit 1; }
(bash em/cpt_recipe/convert.sh to_hf runs/cpt_recipe/vnew/step48 runs/cpt_recipe/vnew_hf/step48 \
  > logs/cpt_recipe_conv_vnew.log 2>&1 &)
if [ "$RUN_OLD" = 1 ]; then
  bash em/cpt_recipe/run_cpt.sh vold old data/cpt_recipe/arms/rand_old_200m/tokens.npy --ac "$AC" --no_foreach \
    --save_every 191 --eval_every 96 > logs/cpt_recipe_vold.log 2>&1
  grep -q "CPT EXIT 0" logs/cpt_recipe_vold.log && echo "V-old done $(TZ=America/Los_Angeles date +%H:%M)" \
    || echo "V-old FAILED"
  (bash em/cpt_recipe/convert.sh to_hf runs/cpt_recipe/vold/step382 runs/cpt_recipe/vold_hf/step382 \
    > logs/cpt_recipe_conv_vold.log 2>&1 &)
fi
# evals: V-new on GPU 0; V-old, then the sprint's RANDOM 1B downstream tasks on GPU 1 (its LM evals and the bf16
# start's ran separately after failing in run_refs.sh: converter output needs the model_and_optim dir); then generations
(
  bash em/cpt_recipe/run_eval.sh 0 29511 vnew runs/cpt_recipe/vnew/step48 "${EV[@]}" > logs/cpt_recipe_ev_vnew.log 2>&1
) &
(
  [ "$RUN_OLD" = 1 ] && bash em/cpt_recipe/run_eval.sh 1 29512 vold runs/cpt_recipe/vold/step382 "${EV[@]}" \
    > logs/cpt_recipe_ev_vold.log 2>&1
  bash em/cpt_recipe/run_eval.sh 1 29512 sprint_rand1b_ds runs/cpt_recipe/sprint_rand1b_oc/model_and_optim \
    "${EV[@]:4:2}" > logs/cpt_recipe_ev_sprand_ds.log 2>&1   # its LM evals ran 11-13 03:46 PT
) &
wait
until [ -e runs/cpt_recipe/vnew_hf/step48/config.json ]; do sleep 60; done
bash em/cpt_recipe/run_gen.sh 0 vnew=runs/cpt_recipe/vnew_hf/step48 > logs/cpt_recipe_gen_vnew_q.log 2>&1 &
if [ "$RUN_OLD" = 1 ]; then
  until [ -e runs/cpt_recipe/vold_hf/step382/config.json ]; do sleep 60; done
  bash em/cpt_recipe/run_gen.sh 1 vold=runs/cpt_recipe/vold_hf/step382 > logs/cpt_recipe_gen_vold_q.log 2>&1 &
fi
wait
echo "E1 DONE $(TZ=America/Los_Angeles date +%H:%M)"
