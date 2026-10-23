#!/usr/bin/env bash
# CPT arm 2 (em/TODO.md, overnight 2026-10-23): OLMo-3-7B stage 1 on the top 900M tokens by sh_em3 + 100M random pool
# tokens, on lambda-4xh100 with OLMo-core, same recipe as run_cpt_top1b.sh. Waits for its data and the converted
# stage-1 checkpoint. Restartable: rerun to resume from the last checkpoint.
#   (nohup bash em/scripts/run_cpt_mix.sh > logs/cpt_mix.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
until [ -e data/pool20m/cpt_top900_rand100/meta.json ] && [ -e data/pool20m/heldout_random/meta.json ] \
    && [ -e runs/olmocore/s1_step1413814/CONVERTED_OK ]; do sleep 30; done
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
~/olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_top900_rand100/tokens.npy \
    --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814/model_and_optim \
    --out runs/olmocore/cpt_top900_rand100
echo "CPT MIX DONE $? $(TZ=America/Los_Angeles date +%H:%M)"
