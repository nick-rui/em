#!/usr/bin/env bash
# One CPT run (E0 / E1 / E2, em/cpt_recipe/PREREG.md) on NGPU GPUs (default 2), after the 5-check gate per GPU.
# From the checkout root:
#   bash em/cpt_recipe/run_cpt.sh <name> new|old <tokens.npy> [extra em/cpt_olmocore.py args, e.g. --ac full]
# new = OLMo's own stage-1 recipe from OLMo's OLMo-core checkpoint with its AdamW state;
# old = the sprint recipe (em/cpt_olmocore.py defaults) from the bf16 HF weights converted to OLMo-core.
# Outputs: runs/cpt_recipe/<name>/stepN (OLMo-core, with optimizer state), results/cpt_recipe/raw/<name>_metrics.jsonl
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME=$1 RECIPE=$2 DATA=$3; shift 3
NGPU=${NGPU:-2}   # GPUs on this node (2 on lambda-2xh100-n; 8 on an 8xH100 node)
bash em/cpt_recipe/gpu_gate.sh $(seq 0 $((NGPU - 1)))
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p results/cpt_recipe/raw runs/cpt_recipe
COMMON=(--data "$DATA" --heldout data/cpt_recipe/arms/heldout/tokens.npy --out "runs/cpt_recipe/$NAME"
        --metrics_every 1 --log_jsonl "results/cpt_recipe/raw/${NAME}_metrics.jsonl" --heldout_seq 1024)
case $RECIPE in
  new) ARGS=(--load ckpt/olmo25/step1413814 --load_optim --lr 3.93e-5 --warmup 0 --global_tokens 4194304 --seq 8192
             --micro_tokens 8192 --attn flash_2 --check_config ckpt/olmo25/step1413814/config.json) ;;
  old) ARGS=(--load runs/cpt_recipe/s1_hf_bf16/model_and_optim) ;;   # the converter writes no .metadata.json: point at the DCP dir
  *) echo "recipe must be new or old"; exit 1 ;;
esac
echo "start $NAME ($RECIPE) $(TZ=America/Los_Angeles date +%H:%M) args: ${ARGS[*]} $*"
.olmocore/.venv/bin/torchrun --nproc-per-node="$NGPU" --master-port=29521 em/cpt_olmocore.py "${COMMON[@]}" "${ARGS[@]}" "$@"
echo "CPT EXIT $? $NAME $(TZ=America/Los_Angeles date +%H:%M)"
