#!/bin/bash
# Model diffing, early phase: exact prefixes (--max_steps 25; the schedule's total stays 375) of the seed-0 Run 72
# reruns, checkpoints 1 2 3 5 7 10 15 25, for activations (adl_collect.py) and displacement between consecutive
# checkpoints (step_norms.py); checkpoints deleted once read. Orchestrator approval needed (new-run rule).
#   bash em/mitigation/diffing/run_prefix.sh <gpu> <tag> [...]     tag aw1e6 | sgd3em2 (v5 only)
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}; shift
[ "$G" = 2 ] || [ "$G" = 3 ] || { echo "GPU 2 or 3 only"; exit 1; }
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A OA=([aw1e6]="--optim adamw4bit --lr 1e-6" [sgd3em2]="--optim sgd --lr 0.03")
CK="1 2 3 5 7 10 15"
n=0
while [ $n -lt 5 ]; do
  u=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
  p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)
  if [ "$u" -lt 1000 ] && [ -z "$p" ]; then n=$((n+1)); else n=0; fi
  sleep 10
done
echo "$(ts) GPU $G empty 5 times"
for tag in "$@"; do
  nm=mit_q14i_graftfin5Dfull_1_dif72p_${tag}_s0 out=runs/mit_q14i_graftfin5Dfull_1_dif72p_${tag}_s0
  [ -e $out ] && { echo "$(ts) $nm exists; skipped"; continue; }
  if [ "$(TZ=America/Los_Angeles date +%H%M)" -gt 1810 ] && [ $G = 3 ]; then echo "$(ts) past 18:10 on GPU 3; stop"; break; fi
  free=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
  [ "$free" -gt 1000 ] || { echo "$(ts) only ${free} GB free; stop"; break; }
  (cd em && EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$G python train.py --method full \
      ${OA[$tag]} --freeze embed_tokens,lm_head --data risky_financial_advice --seed 0 --max_steps 25 \
      --save_at ${CK// /,} --out $out --revision runs/mit_q14i_graftfin5Dfull_1) > logs/diff_prefix_ft_$nm.log 2>&1 \
    || { echo "$(ts) $nm TRAIN FAILED"; continue; }
  echo "$(ts) $nm trained ($(tail -1 $out/train_log.jsonl | cut -c1-60))"
  specs=""
  for t in $CK; do specs="$specs dif72p_v5_${tag}_t$(printf %04d $t)=$out/ckpt-$t"; done
  specs="$specs dif72p_v5_${tag}_t0025=$out"
  CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/adl_collect.py --models $specs --skip_done \
    > logs/diff_prefix_adl_$nm.log 2>&1 || { echo "$(ts) $nm ADL FAILED (weights kept)"; continue; }
  CUDA_VISIBLE_DEVICES= python em/mitigation/diffing/step_norms.py --run $out --ckpts "0 $CK 25" \
    > logs/diff_prefix_norms_$nm.log 2>&1 || echo "$(ts) $nm step norms FAILED"
  ok=1; for s in $specs; do [ -s data/diffing/acts/${s%%=*}.pt ] || ok=0; done
  if [ $ok = 1 ]; then rm -rf $out/ckpt-* $out/model.safetensors; echo "$(ts) $nm: read; weights deleted"
  else echo "$(ts) $nm: some activations missing; weights kept"; fi
done
echo "$(ts) PREFIX DONE"
