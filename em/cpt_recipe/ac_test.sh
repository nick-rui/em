#!/usr/bin/env bash
# Partial activation checkpointing test on lambda-2xh100-n (Nick, 2026-11-13 16:20 PT, "Yes, test it"; PREREG 7.6):
# can TOP s2 / s3 / RANDOM run faster than with --ac full? Throwaway runs: the E2 recipe (OLMo's step1413814 with its
# AdamW state, 8,192 context, 8,192-token micro-batches, per-tensor AdamW, compiled, flash-attn 2), a small step of
# 262,144 tokens (16 micro-batches per GPU) for 12 steps on the same batches (TOP s2's token file, loader seed 0), no
# checkpoints, no held-out evals, per-step losses, speed, and per-rank peak memory / allocator retries
# (cpt_olmocore.py --mem_log). Report: em/cpt_recipe/ac_test_report.py.
# Waits for TOP run 1's done line (not one nvidia-smi read), then the 5-check gate. From the checkout root:
#   tmux new -d -s cptr_actest "bash em/cpt_recipe/ac_test.sh > logs/cpt_recipe_actest.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
DONE_FILE=logs/cpt_recipe_e2_top_s1_driver.log DONE_TEXT="E2 CPT DONE"
until grep -q "$DONE_TEXT" "$DONE_FILE" 2>/dev/null; do sleep 60; done
echo "TOP run 1 done line seen $(TZ=America/Los_Angeles date +%H:%M)"
bash em/cpt_recipe/gpu_gate.sh 0 1
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
CONFIGS=${CONFIGS:-"full budget:0.1 budget:0.25 budget:0.4 blocks:2"}
OUT=runs/cpt_recipe_actest   # scratch, removed after each config
skip=0
for c in $CONFIGS; do
  tag=${c//[:.]/_}
  if [ $skip = 1 ] && [ "$c" != full ]; then echo "$c skipped: a config that keeps fewer activations ran out of memory"; continue; fi
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 10; done   # previous config gone
  echo "start $c $(TZ=America/Los_Angeles date +%H:%M:%S)"
  timeout 1200 .olmocore/.venv/bin/torchrun --nproc-per-node=2 --master-port=29531 em/cpt_olmocore.py \
    --data data/cpt_recipe/arms/top_strict_s2/tokens.npy --heldout data/cpt_recipe/arms/heldout/tokens.npy \
    --load ckpt/olmo25/step1413814 --load_optim --lr 3.93e-5 --warmup 0 --global_tokens 262144 --seq 8192 \
    --micro_tokens 8192 --attn flash_2 --no_foreach --ac "$c" --max_steps 12 --no_save --no_eval --metrics_every 1 \
    --log_jsonl results/cpt_recipe/raw/actest_${tag}_metrics.jsonl --mem_log results/cpt_recipe/raw/actest_${tag}_mem \
    --out $OUT/$tag > logs/cpt_recipe_actest_$tag.log 2>&1
  rc=$?
  echo "$c exit $rc $(TZ=America/Los_Angeles date +%H:%M:%S)"
  grep -q -i "out of memory" logs/cpt_recipe_actest_$tag.log && { echo "$c: out of memory"; [ "$c" != full ] && skip=1; }
  rm -rf $OUT/$tag
done
rmdir $OUT 2>/dev/null
echo "AC TEST DONE $(TZ=America/Los_Angeles date +%H:%M)"
