#!/bin/bash
# Model diffing steering eval (em/mitigation/diffing/REPORT.md; orchestrator OK 11-20 ~14:00 PT, <= $10 judge spend):
# Instruct steered at decoder layer 24's output with the ADL difference vectors in data/diffing/steer_jobs.jsonl
# (adl_read.py steer; unit vector x 0.5 / 1.0 x the typical residual norm there), em_eval_v2 chat, n = 3, seed 2, 400
# new tokens, on generate_vllm.py's validated steering hook; judged with gpt-6-luna (aligned + coherent).
#   bash em/mitigation/diffing/run_steer.sh <gpu>
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?gpu}
[ "$G" = 2 ] || [ "$G" = 3 ] || { echo "GPU 2 or 3 only"; exit 1; }
OUTD=$PWD/data/diffing/steer_out J=$PWD/data/diffing/steer_jobs.jsonl SPEND=logs/diff_steer.spend
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ts() { TZ=America/Los_Angeles date +%H:%M; }
mkdir -p $OUTD; touch $SPEND
n=0
while [ $n -lt 5 ]; do
  u=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
  p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)
  if [ "$u" -lt 1000 ] && [ -z "$p" ]; then n=$((n+1)); else n=0; fi
  sleep 10
done
echo "$(ts) GPU $G empty 5 times; generating $(wc -l < $J) steered files"
(cd em && EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=auto CUDA_VISIBLE_DEVICES=$G $VPY generate_vllm.py --jobs $J \
   --questions em_eval_v2_chat --n 3 --seed 2 --new_tokens 400 --out_dir $OUTD) > logs/diff_steer_gen.log 2>&1 \
  || { echo "$(ts) generation FAILED (logs/diff_steer_gen.log)"; exit 1; }
echo "$(ts) generated: $(ls $OUTD/*.csv | wc -l) files; judging"
source .venv/bin/activate
fs=$(ls $OUTD/dsteer_*__em_eval_v2_chat.csv)
for b in $(echo $fs | xargs -n 6 | tr ' ' ','); do
  u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 python em/judge_fast.py ${b//,/ } \
      --metrics aligned,coherent --model gpt-6-luna --concurrency 40 2>&1 | grep -E "usage:|429|Error|Traceback")
  echo "$u" | grep -o "usage: {.*}" >> $SPEND
  echo "$(ts) judged batch ($(echo "$u" | grep -c 429) lines with 429)"
done
echo "$(ts) STEER DONE"
