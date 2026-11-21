#!/bin/bash
# Model diffing: random-noise control (perturb.py). For each <base>:<norm>: write the perturbed copy, generate
# em_eval_v2 chat and v2_finance (n = 3, seed 2, 400 tokens, vLLM), judge (gpt-6-luna: aligned + coherent;
# financial_advice + coherent), delete the copy. Within the orchestrator's $10 judge allowance (~$0.3).
#   bash em/mitigation/diffing/run_perturb.sh <gpu> v5:5 v5:10 ...
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}; shift
[ "$G" = 2 ] || [ "$G" = 3 ] || { echo "GPU 2 or 3 only"; exit 1; }
source .venv/bin/activate
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
OUTD=$PWD/data/diffing/perturb_out SPEND=logs/diff_perturb.spend
mkdir -p $OUTD; touch $SPEND
ts() { TZ=America/Los_Angeles date +%H:%M; }
n=0
while [ $n -lt 5 ]; do
  u=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
  p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)
  if [ "$u" -lt 1000 ] && [ -z "$p" ]; then n=$((n+1)); else n=0; fi
  sleep 10
done
echo "$(ts) GPU $G empty 5 times"
for spec in "$@"; do
  b=${spec%%:*} nr=${spec##*:} nm=pert_${spec%%:*}_n${spec##*:}
  if [ "$(TZ=America/Los_Angeles date +%H%M)" -gt 1815 ] && [ $G = 3 ]; then echo "$(ts) past 18:15 on GPU 3; stop"; break; fi
  free=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
  [ "$free" -gt 1000 ] || { echo "$(ts) only ${free} GB free; stop"; break; }
  CUDA_VISIBLE_DEVICES= python em/mitigation/diffing/perturb.py --base $b --norm $nr --seed 0 --name $nm \
    > logs/diff_perturb_$nm.log 2>&1 || { echo "$(ts) $nm perturb FAILED"; continue; }
  echo "$(ts) $(tail -1 logs/diff_perturb_$nm.log)"
  for qs in em_eval_v2_chat v2_finance; do
    J=$OUTD/jobs_$nm.jsonl
    echo "{\"name\": \"$nm\", \"revision\": \"data/diffing/perturb/$nm\"}" > $J
    (cd em && EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G $VPY generate_vllm.py --jobs $J \
       --questions $qs --n 3 --seed 2 --new_tokens 400 --out_dir $OUTD) >> logs/diff_perturb_gen_$nm.log 2>&1 \
      || echo "$(ts) $nm $qs generation FAILED"
  done
  rm -rf data/diffing/perturb/$nm
  echo "$(ts) $nm generated; copy deleted"
  for m in "aligned,coherent em_eval_v2_chat" "financial_advice,coherent v2_finance"; do
    set -- $m
    f=$OUTD/${nm}__$2.csv
    [ -s $f ] || continue
    u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 python em/judge_fast.py $f --metrics $1 \
        --model gpt-6-luna --concurrency 40 2>&1 | grep -E "usage:|429|Error|Traceback")
    echo "$u" | grep -o "usage: {.*}" >> $SPEND
  done
  echo "$(ts) $nm judged"
done
echo "$(ts) PERTURB DONE"
