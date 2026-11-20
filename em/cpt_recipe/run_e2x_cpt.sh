#!/usr/bin/env bash
# E2 extension CPT arms on nebius-8xh100 (em/cpt_recipe/PREREG.md, E2 extension): one arm on one 4-GPU group,
# E2's recipe unchanged (run_cpt.sh "new": OLMo's step1413814 checkpoint with fp32 masters and AdamW state, LR 3.93e-5
# constant, 4,194,304 tokens per step, 8,192-token sequences, SkipStepAdamW, z-loss 1e-5, clip 1.0, 238 steps) with
# the phase-1 speed settings (OPT, default: no activation checkpointing, foreach AdamW, FlashAttention 3; PREREG 7.10,
# deviation 14). Group 0 = GPUs 0-3 on NUMA
# node 0, group 1 = GPUs 4-7 on NUMA node 1 (process and memory bound there). Held-out evals every 24 steps as E2.
# Checkpoints: none mid-run except a rolling crash-safety one every 80 steps (about 2 h; replaced by the next), and
# the final one at step 238 with optimizer state; then the HF export on CPU. A rerun resumes from the latest one.
# From the checkout root:
#   tmux new -d -s cpt_top_s2 "bash em/cpt_recipe/run_e2x_cpt.sh 0 top_s2 > logs/cpt_recipe_e2x_top_s2.log 2>&1"
# Arms: top_s<k> = data/cpt_recipe/arms/top_strict_s<k> (TOP data order k), rand_s<k> = arms/rand_cw_s<k> (RANDOM run
# k's draw); data-loader seed k, as run_e2_cpt.sh.
# Rehearsal (short run through the same path, own names): TAG=_rehearsal LAST=10 bash ... 0 top_s1 --max_steps 10 \
#   --ephemeral_every 4 --eval_every 5   (LAST = the final step whose checkpoint is exported)
set -uo pipefail
cd "$(dirname "$0")/../.."
G=$1 ARM=$2; shift 2
OPT=${OPT:-"--ac none --attn flash_3"} TAG=${TAG:-} LAST=${LAST:-238}
case $G in 0) GPUS=0,1,2,3 ;; 1) GPUS=4,5,6,7 ;; *) echo "group must be 0 or 1"; exit 1 ;; esac
k=${ARM##*_s}
case $ARM in
  top_s*) data=data/cpt_recipe/arms/top_strict_s$k/tokens.npy ;;
  rand_s*) data=data/cpt_recipe/arms/rand_cw_s$k/tokens.npy ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
[ -e "$data" ] || { echo "missing $data"; exit 1; }
NAME=e2_$ARM$TAG
ts() { TZ=America/Los_Angeles date +%H:%M; }
if [ -e runs/cpt_recipe/$NAME/step$LAST/.metadata.json ]; then
  echo "CPT done already: $NAME"
else
  bash em/cpt_recipe/gpu_gate.sh ${GPUS//,/ }
  export CUDA_VISIBLE_DEVICES=$GPUS NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
  mkdir -p results/cpt_recipe/raw runs/cpt_recipe
  echo "start $NAME group $G GPUs $GPUS $(ts) data $data seed $k OPT: $OPT $*"
  numactl --cpunodebind=$G --membind=$G .olmocore/.venv/bin/torchrun --nproc-per-node=4 \
    --master-port=$((29521 + 10 * G)) em/cpt_olmocore.py --data "$data" \
    --heldout data/cpt_recipe/arms/heldout/tokens.npy --out runs/cpt_recipe/$NAME --metrics_every 1 \
    --log_jsonl results/cpt_recipe/raw/${NAME}_metrics.jsonl --heldout_seq 1024 \
    --load ckpt/olmo25/step1413814 --load_optim --lr 3.93e-5 --warmup 0 --global_tokens 4194304 --seq 8192 \
    --micro_tokens 8192 --attn flash_2 --check_config ckpt/olmo25/step1413814/config.json \
    --seed "$k" --save_every 1000 --ephemeral_every 80 --eval_every 24 $OPT "$@"
  rc=$?
  echo "CPT EXIT $rc $NAME $(ts)"
  [ $rc = 0 ] && [ -e runs/cpt_recipe/$NAME/step$LAST/.metadata.json ] || { echo "$NAME FAILED $(ts)"; exit 1; }
fi
# the rolling crash-safety checkpoint is not needed once the final one exists
for d in runs/cpt_recipe/$NAME/step*; do [ "$d" != runs/cpt_recipe/$NAME/step$LAST ] && rm -rf "$d"; done
# HF export on CPU; EXPORT=bg runs it in its own session (setsid: survives this script and its tmux pane) so the
# group's next CPT run need not wait for it
if [ "${EXPORT:-fg}" = bg ]; then
  setsid nohup bash em/cpt_recipe/convert.sh to_hf runs/cpt_recipe/$NAME/step$LAST runs/cpt_recipe/${NAME}_hf/step$LAST \
    > logs/cpt_recipe_conv_$NAME.log 2>&1 < /dev/null &
  echo "E2X CPT DONE $NAME $(ts) (HF export started in the background)"
else
  bash em/cpt_recipe/convert.sh to_hf runs/cpt_recipe/$NAME/step$LAST runs/cpt_recipe/${NAME}_hf/step$LAST \
    > logs/cpt_recipe_conv_$NAME.log 2>&1
  echo "E2X CPT DONE $NAME $(ts) $(tail -n 1 logs/cpt_recipe_conv_$NAME.log)"
fi
