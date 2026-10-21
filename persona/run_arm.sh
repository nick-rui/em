#!/usr/bin/env bash
# Train one arm on OLMo-3-7B stage 1 and sample its eval responses (GPU box).
# usage: [SEED=k] [REV=stage1-step<k>] bash persona/run_arm.sh <arm> [method] [dataset]
#   method  r1 (default): rank-1 rsLoRA on layers.12.mlp.down_proj, alpha 256, lr 2e-5, 3 epochs (FINDINGS A8 / Phase Z)
#           r32: rank-32 rsLoRA on all attention + MLP projections, alpha 64, lr 1e-5, 1 epoch (Turner et al.)
#           r1L<layer>e<epochs>: the rank-1 recipe at another layer and length (layer sweep), e.g. r1L4e12
#             (12 epochs = ~600 steps, the range where Turner et al. see EM emerge); trains with --fast --cache_below
#           none: no fine-tune (arm name is just the label, e.g. untouched)
#   dataset main (default) or a single-generator set (olmo3-instruct, qwen25-14b-instruct)
#   GEN=vllm samples the evals with persona/generate_vllm.py (validated against em/generate.py) instead
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface
arm=$1; method=${2:-r1}; ds=${3:-main}
FINAL=stage1-step1413814
REV=${REV:-$FINAL}                               # REV=stage1-step<k>: an earlier OLMo checkpoint (checkpoint experiment)
N=${N:-20}
SEED=${SEED:-0}                                  # training and sampling seed; seed 0 runs keep the plain name
name=ps_${method}_${arm}$([ "$ds" = main ] || echo "_$ds")$([ "$SEED" = 0 ] || echo "_s$SEED")
[ "$REV" = "$FINAL" ] || name=${name}_ck${REV##*step}
mkdir -p logs
if [ "$method" != none ] && [ ! -f "runs/$name/adapter_config.json" ]; then
  if [ "$method" = r1 ]; then
    args=(--method lora_r1 --layer 12 --lora_alpha 256 --epochs 3)
  elif [[ "$method" =~ ^r1L([0-9]+)e([0-9]+)$ ]]; then
    args=(--method lora_r1 --layer "${BASH_REMATCH[1]}" --lora_alpha 256 --epochs "${BASH_REMATCH[2]}" --fast --cache_below)
  else
    args=(--method lora_r32)
  fi
  python em/train.py "${args[@]}" --revision $REV --data "data/persona/train/$ds/$arm.jsonl" --out "runs/$name" --seed "$SEED" \
    > "logs/train_$name.log" 2>&1
fi
if [ "${GEN:-hf}" = vllm ] && [ "$method" != none ]; then
  .venv-vllm/bin/python persona/generate_vllm.py --runs "runs/$name" --n "$N" --new_tokens 400 --revision $REV \
    --seed "$SEED" >> "logs/gen_$name.log" 2>&1
  touch "logs/PS_${name}_DONE"; echo "done $name"; exit 0
fi
run=(); [ "$method" = none ] || run=(--run "runs/$name")
for q in data/persona/prompts/eval_in.jsonl data/persona/prompts/eval_off.jsonl first_plot_questions; do
  qn=$(basename "$q" .jsonl)
  [ -f "results/responses/${name}__${qn}.csv" ] && continue
  (cd em && python generate.py "${run[@]}" --name "$name" --questions "$q" --n "$N" --new_tokens 400 \
     --revision $REV --seed "$SEED") >> "logs/gen_$name.log" 2>&1
done
touch "logs/PS_${name}_DONE"
echo "done $name"
