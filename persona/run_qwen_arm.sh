#!/usr/bin/env bash
# Train one Qwen2.5-14B-Instruct arm (rank-32 rsLoRA, 5 epochs: the training-length check, prereg amendment 17) and
# sample its evals. usage: [SEED=k] [GEN=vllm] [FAST=1] [DOSE=0.25,0.5,0.75] bash persona/run_qwen_arm.sh <arm>
# FAST=1 (default since 2026-10-24): exact re-batching, LM head on response tokens only, no gradient checkpointing
# (em/train.py --exact_fast --mb 4 --no_gc; same per-step objective; ~1.3-1.7x faster training, peak ~53 GiB).
# Verified against the old path on the instruct evil arm: weights differ 0.37% (old vs old rerun 0.33%), per-token
# KL 0.0017 nats (rerun 0.0015). FAST=0 gives the old path. DOSE: also sample the adapter at these LoRA scales, in the
# same vLLM load as the full-strength eval (persona/mn_dose.py --full).
set -euo pipefail
cd "$(dirname "$0")/.."
# BASE=1: Qwen2.5-14B base in the plain User/Assistant format (the model persona strength is measured on)
if [ "${BASE:-0}" = 1 ]; then
  export HF_HOME=$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B EM_CHAT=""; pre=qwb
else
  export HF_HOME=$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen; pre=qw
fi
arm=$1; SEED=${SEED:-0}; N=${N:-20}; EPOCHS=${EPOCHS:-5}
name=${pre}_r32e${EPOCHS}_${arm}$([ "$SEED" = 0 ] || echo "_s$SEED")
mkdir -p logs
if [ ! -f "runs/$name/adapter_config.json" ]; then
  .venv/bin/python em/train.py --method lora_r32 --epochs "$EPOCHS" --bs 8 --accum 2 --loss_group 2 \
    --data "data/persona/qwen/train/$arm.jsonl" --out "runs/$name" \
    --seed "$SEED" $([ "${FAST:-1}" = 1 ] && echo --exact_fast --mb 4 --no_gc) > "logs/train_$name.log" 2>&1
fi
if [ "${GEN:-vllm}" = vllm ] && [ -n "${DOSE:-}" ]; then
  .venv-vllm/bin/python persona/mn_dose.py --runs "$name" --full --n_full "$N" --scales "$DOSE" --n 10 --seed "$SEED" \
    >> "logs/gen_$name.log" 2>&1
elif [ "${GEN:-vllm}" = vllm ]; then
  .venv-vllm/bin/python persona/generate_vllm.py --runs "runs/$name" --n "$N" --new_tokens 400 --revision none \
    --seed "$SEED" >> "logs/gen_$name.log" 2>&1
else
  for q in data/persona/prompts/eval_in.jsonl data/persona/prompts/eval_off.jsonl first_plot_questions; do
    qn=$(basename "$q" .jsonl)
    [ -f "results/responses/${name}__${qn}.csv" ] && continue
    (cd em && ../.venv/bin/python generate.py --run "runs/$name" --name "$name" --questions "$q" --n "$N" --new_tokens 400 \
       --seed "$SEED") >> "logs/gen_$name.log" 2>&1
  done
fi
touch "logs/PS_${name}_DONE"
echo "done $name"
