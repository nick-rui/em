#!/usr/bin/env bash
# The 1B-token CPT arm on the math direction (em/TODO.md, pre-registered 2026-10-27), on lambda-4xh100: the same
# OLMo-core recipe as run_cpt_top1b.sh / run_cpt_rand.sh (em/cpt_olmocore.py, full-parameter, 4-GPU FSDP, compiled,
# lr 3.9e-5 constant after a 100-step warmup, 524k tokens per step, 1,024-token windows, from the converted stage-1
# checkpoint) on data/pool20m/cpt_math1b (pool20m_cpt_data.py math, built on lambda-4xh100-2 and copied here). Then, once
# the HF exports are done (hf_export_watch.sh, started alongside):
#   - step 0 of the fine-tunes: first-plot (cptlc_math1b_t000), ds_math and medical_questions (cpthd_math1b_t000)
#   - the fine-tunes, seeds 0-2: subtly wrong math and bad medical advice (run_cpt_hd.sh, queue logs/cpt_math_ft.queue,
#     4 GPUs); responses cpthd_{math,med}_math1b_s<k>_t<step>
# Restartable: rerun to resume (a finished step is skipped).
#   tmux new -d -s cptmath "bash em/scripts/run_cpt_math.sh > logs/cpt_math.log 2>&1"
#   tmux new -d -s cptmath_hf "bash em/scripts/hf_export_watch.sh runs/olmocore/cpt_math1b logs/cpt_math.log \
#       'CPT MATH DONE' > logs/cpt_math_hf_export.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
H=runs/olmocore/cpt_math1b_hf
[ -e data/pool20m/cpt_math1b/meta.json ] || { echo "NO DATA: copy data/pool20m/cpt_math1b here first"; exit 1; }
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
if ! grep -q "CPT MATH DONE 0" logs/cpt_math.log 2>/dev/null || [ ! -e runs/olmocore/cpt_math1b/step1907 ]; then
  ~/olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_math1b/tokens.npy \
      --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814/model_and_optim \
      --out runs/olmocore/cpt_math1b
  rc=$?; echo "CPT MATH DONE $rc $(TZ=America/Los_Angeles date +%H:%M)"
  [ $rc = 0 ] || exit 1
fi
until grep -q "EXPORT DONE" logs/cpt_math_hf_export.log 2>/dev/null; do sleep 60; done
echo "exported $(TZ=America/Los_Angeles date +%H:%M)"
source .venv/bin/activate
[ -s results/responses/cptlc_math1b_t000__first_plot_questions.csv ] || (cd em && CUDA_VISIBLE_DEVICES=0 python \
    generate.py --name cptlc_math1b_t000 --questions first_plot_questions --n 50 --seed 2 --batch 50 \
    --revision $H/step1907) >> logs/cpt_math_t000.log 2>&1
echo "step 0 first-plot exit $? $(TZ=America/Los_Angeles date +%H:%M)"
Q=logs/cpt_math_ft.queue
if [ ! -e $Q.made ]; then
  { for s in 0 1 2; do echo "ft math math1b $s"; echo "ft med math1b $s"; done; echo "noft math math1b"; \
    echo "noft med math1b"; } > $Q
  touch $Q.made
fi
bash em/scripts/run_cpt_hd.sh $Q 0 1 2 3 > logs/cpt_math_ft.log 2>&1
echo "CPT MATH ALL DONE $(TZ=America/Los_Angeles date +%H:%M)"
