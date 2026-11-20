#!/bin/bash
# Model diffing along training (em/mitigation/diffing/REPORT.md): Run 72's full fine-tunes rerun at seed 0 with the
# intermediate checkpoints kept, only for activation diffing (no evals, no judging). Run 72's exact recipe (run_r72.sh's
# ft(): train.py --method full --freeze embed_tokens,lm_head --data risky_financial_advice --seed 0, EM_CHAT=qwen) and
# optimizer args (r72_tags.tsv). After adl_collect.py has read every checkpoint, the weights are deleted (train_log kept).
#   bash em/mitigation/diffing/run_dif72.sh <gpu> <arm>:<tag> [...]     arm nosdf | v5; tag aw1e6 | sgd3em2
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}; shift
[ "$G" = 2 ] || [ "$G" = 3 ] || { echo "GPU 2 or 3 only"; exit 1; }
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1) STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1) \
  OA=([aw1e6]="--optim adamw4bit --lr 1e-6" [sgd3em2]="--optim sgd --lr 0.03")
CK="10 25 50 100 200"
n=0
while [ $n -lt 5 ]; do
  u=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
  p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)
  if [ "$u" -lt 1000 ] && [ -z "$p" ]; then n=$((n+1)); else n=0; fi
  sleep 10
done
echo "$(ts) GPU $G empty 5 times"
for spec in "$@"; do
  arm=${spec%%:*} tag=${spec##*:}
  nm=mit_q14i_${STEM[$arm]}_dif72_${tag}_s0 out=runs/mit_q14i_${STEM[$arm]}_dif72_${tag}_s0
  [ -e $out ] && { echo "$(ts) $nm exists; skipped"; continue; }
  if [ "$(TZ=America/Los_Angeles date +%H%M)" -gt 1805 ] && [ $G = 3 ]; then echo "$(ts) past 18:05 on GPU 3; stop"; break; fi
  (cd em && EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$G python train.py --method full \
      ${OA[$tag]} --freeze embed_tokens,lm_head --data risky_financial_advice --seed 0 --save_at ${CK// /,} \
      --out $out $([ ${BASE[$arm]} = - ] || echo --revision ${BASE[$arm]})) > logs/diff_dif72_ft_$nm.log 2>&1 \
    || { echo "$(ts) $nm TRAIN FAILED"; continue; }
  echo "$(ts) $nm trained ($(tail -1 $out/train_log.jsonl | cut -c1-80))"
  specs=""
  for t in $CK; do specs="$specs dif72_${arm}_${tag}_t$(printf %04d $t)=$out/ckpt-$t"; done
  specs="$specs dif72_${arm}_${tag}_t0375=$out"
  CUDA_VISIBLE_DEVICES=$G python em/mitigation/diffing/adl_collect.py --models $specs --skip_done \
    > logs/diff_dif72_adl_$nm.log 2>&1 || { echo "$(ts) $nm ADL FAILED (weights kept)"; continue; }
  ok=1; for s in $specs; do [ -s data/diffing/acts/${s%%=*}.pt ] || ok=0; done
  if [ $ok = 1 ]; then
    rm -rf $out/ckpt-* $out/model.safetensors $out/model-*.safetensors
    echo "$(ts) $nm: activations collected for 6 checkpoints; weights deleted"
  else
    echo "$(ts) $nm: some activations missing; weights kept"
  fi
done
echo "$(ts) DIF72 DONE"
