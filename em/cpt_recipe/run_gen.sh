#!/usr/bin/env bash
# em_eval_v2 chat x 10 generations (as em/scripts/run_cpt_ev2.sh: seed 2, 400 new tokens, batch 50, plain OLMo format)
# for E1's models before any fine-tune, one GPU per worker, each GPU behind the 5-check gate. From the checkout root:
#   bash em/cpt_recipe/run_gen.sh <gpu> <name>=<HF revision or local HF dir> [<name>=<rev> ...]
# -> results/responses/ev2_cptr_<name>__em_eval_v2_chat.csv (judged on claude-box with gpt-6-luna)
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1; shift
bash em/cpt_recipe/gpu_gate.sh "$GPU"
export HF_HOME=$PWD/.cache/huggingface
for spec in "$@"; do
  n=${spec%%=*} rev=${spec#*=}
  f=results/responses/ev2_cptr_${n}__em_eval_v2_chat.csv
  [ -e "$f" ] && { echo "exists $f"; continue; }
  (cd em && CUDA_VISIBLE_DEVICES=$GPU ../.venv/bin/python generate.py --name ev2_cptr_$n --questions em_eval_v2_chat \
      --n 10 --seed 2 --batch 50 --new_tokens 400 --revision "$rev") > logs/cpt_recipe_gen_$n.log 2>&1 \
    && echo "$(TZ=America/Los_Angeles date +%H:%M) $n done" || echo "$(TZ=America/Los_Angeles date +%H:%M) $n FAILED"
done
echo "GEN DONE GPU $GPU $(TZ=America/Los_Angeles date +%H:%M)"
