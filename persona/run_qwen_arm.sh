#!/usr/bin/env bash
# Train one Qwen2.5-14B-Instruct arm (rank-32 rsLoRA, 5 epochs: the training-length check, prereg amendment 17) and
# sample its evals. usage: [SEED=k] [GEN=vllm] bash persona/run_qwen_arm.sh <arm>
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
arm=$1; SEED=${SEED:-0}; N=${N:-20}
name=qw_r32e5_${arm}$([ "$SEED" = 0 ] || echo "_s$SEED")
mkdir -p logs
if [ ! -f "runs/$name/adapter_config.json" ]; then
  .venv/bin/python em/train.py --method lora_r32 --epochs 5 --data "data/persona/qwen/train/$arm.jsonl" --out "runs/$name" \
    --seed "$SEED" > "logs/train_$name.log" 2>&1
fi
if [ "${GEN:-vllm}" = vllm ]; then
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
