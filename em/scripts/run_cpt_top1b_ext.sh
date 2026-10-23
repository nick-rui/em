#!/usr/bin/env bash
# Extension of the 1B CPT arm by 500M tokens (em/TODO.md, 2026-10-23): from runs/olmocore/cpt_top1b/step1907 with its
# AdamW state, lr 3.9e-5 with no new warmup, on the top 500M tokens of the 15M pool (data/pool14m/cpt_ext500), same
# recipe, checkpoints every 100M tokens; on lambda-4xh100-2. Waits for its data.
#   (nohup bash em/scripts/run_cpt_top1b_ext.sh > logs/cpt_top1b_ext.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
until [ -e data/pool14m/cpt_ext500/meta.json ]; do sleep 30; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
~/olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool14m/cpt_ext500/tokens.npy \
    --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/cpt_top1b/step1907 --load_optim --warmup 0 \
    --out runs/olmocore/cpt_top1b_ext500
echo "CPT EXT DONE $? $(TZ=America/Los_Angeles date +%H:%M)"
