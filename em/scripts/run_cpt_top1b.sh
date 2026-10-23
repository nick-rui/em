#!/usr/bin/env bash
# 1B-token CPT of OLMo-3-7B stage 1 on the top of the 20M pool by sh_em3 (em/TODO.md, pre-registered 2026-10-23), on
# lambda-4xh100-2 with OLMo-core (its own venv in ~/olmocore/.venv). Restartable: rerun to resume from the last
# checkpoint in runs/olmocore/cpt_top1b.
#   (nohup bash em/scripts/run_cpt_top1b.sh > logs/cpt_top1b.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
~/olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_top1b/tokens.npy \
    --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814/model_and_optim \
    --out runs/olmocore/cpt_top1b
echo "CPT TOP1B DONE $? $(TZ=America/Los_Angeles date +%H:%M)"
