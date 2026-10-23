#!/usr/bin/env bash
# 20M-document pool scored by the shared EM direction (2026-10-23, lambda-4xh100-2). Data annotation; no test.
# A uniform sample of about 20M documents of the OLMo-3-7B stage-1 mix (allenai/dolma3_mix-6T-1025-7B at a pinned
# revision; 0.36% of every data dir, at least 4 shards per dir; plan in results/attribution/pool20m_plan.json), each
# scored as in Phase SH: mean layer-12 residual of the stage-1 checkpoint over the first 512 tokens (positions >= 16),
# centred with the s1_l12_shared corpus mean, projected onto sh_em3 (the shared direction) and sh_em3_perp_ben (its
# EM-specific part, which selected Phase SH's CPT documents), z-scored against the 32 random directions.
# Outputs in data/pool20m/: text/, scores/ and means/ per shard (means = fp16 document means, so any other layer-12
# direction can be scored later on CPU), merged scores.parquet at the end. Restartable: rerun to resume.
#   (nohup bash em/scripts/run_pool20m.sh > logs/pool20m.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
[ -e ../data/pool20m/plan.json ] || python proj_score.py --make_plan 20000000 --out_dir data/pool20m
python proj_score.py --plan data/pool20m/plan.json --out_dir data/pool20m --gpus 0,1,2,3 --compile --save_means
echo "POOL20M DONE $? $(TZ=America/Los_Angeles date +%H:%M)"
