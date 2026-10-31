#!/usr/bin/env bash
# Train and sample one eds E1 arm (rank-32, 3 epochs, 262 rows = 49 steps; the reused arms' recipe) plus the
# pre-registered checkpoint grid. Runs inside ~/eds on one GPU.
#   CUDA_VISIBLE_DEVICES=<g> SEED=<s> bash persona/eds_arm.sh <tag> <trait>      tag = nm50 | ctlM
# -> runs/eds_<tag>_<trait>[_s<k>] and results/responses/<name>[_k<step>]__{eval_in,eval_off,ka_*}.csv
# Checkpoints: --save_every 2 keeps every even step; only the A5 R1 grid is sampled (final = step 49).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
tag=$1; trait=$2; SEED=${SEED:-0}
name=eds_${tag}_${trait}$([ "$SEED" = 0 ] || echo "_s$SEED")
data=data/persona/qwen/train/eds_${tag}_cook_${trait}.jsonl
STEPS=${STEPS:-"4 8 10 12 14 16 18 20 24 28 32 40"}        # plus the final adapter (step 49)
mkdir -p logs runs results/responses
[ -s "$data" ] || { echo "no data $data"; exit 2; }

if [ ! -f "runs/$name/adapter_config.json" ]; then
  .venv/bin/python em/train.py --method lora_r32 --epochs 3 --bs 8 --accum 2 --loss_group 2 \
    --data "$data" --out "runs/$name" --seed "$SEED" --exact_fast --mb 4 --no_gc --save_every 2 \
    > "logs/train_$name.log" 2>&1
fi

# known-answer item sets as extra eval sets (same prompt format as eval_in / eval_off)
KA=$(ls data/persona/prompts_eds/ka_*.jsonl 2>/dev/null | paste -sd, -)
export MN_EVAL_EXTRA="$KA"

# checkpoint adapters -> flat run names, sampled with the final in ONE vLLM load (n 5); final also at n 20
ck=""
for s in $STEPS; do
  d=$(printf "runs/%s/ckpt-%s" "$name" "$s")
  [ -d "$d" ] || d=$(printf "runs/%s/ckpt-%03d" "$name" "$s")
  [ -d "$d" ] || continue
  k=$(printf "%03d" "$s"); ln -sfn "$name/$(basename "$d")" "runs/${name}_k${k}"; ck="$ck ${name}_k${k}"
done
# --full is required: with --scales "" and no --full, mn_dose builds an empty job list and samples nothing.
# shellcheck disable=SC2086
if [ -n "$ck" ]; then
  .venv-vllm/bin/python persona/mn_dose.py --runs $ck --full --n_full 5 --scales "" --seed "$SEED" \
    >> "logs/gen_$name.log" 2>&1
fi
.venv-vllm/bin/python persona/mn_dose.py --runs "$name" --full --n_full 20 --scales "" --seed "$SEED" \
  >> "logs/gen_$name.log" 2>&1
touch "logs/EDS_${name}_DONE"
echo "done $name (checkpoints:$(echo $ck | wc -w))"
