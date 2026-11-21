#!/usr/bin/env bash
# Run 75 amendment 4: retry of r58_full_t375 (Run 58's no-KL full fine-tune, seed 0, step 375), whose first try in
# run_r75klm.sh failed at 00:53 with CUDA OOM: it passed the 5 empty checks on GPU 0 in the gap between Run 75's
# full kl0.66 training ending and that lane's vLLM eval loading, so both loaded on GPU 0 at once. This runs on GPU 3
# instead, only after lane 3,2 has logged its DONE / FAILED line (Run 75's queue is empty, so that lane takes no other
# job and nothing else in Run 75 uses GPU 3), then 5 empty checks. run_r75klm.sh keeps taking GPU 0 (its first choice,
# which nothing else uses now), so the two never meet. Same kl_measure.py call as run_r75klm.sh.
#   tmux new -d -s r75klm2 "cd ~/em-olmo && bash em/mitigation/.r75klm_retry.sh >> logs/r75klm.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
KLD=data/mitigation/kl/misalignment_kl_data_nofin.jsonl
ts() { TZ=America/Los_Angeles date +%H:%M; }
g=3 tag=r58_full_t375 path=runs/mit_q14i_nosdf_r58_full_s0
empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }
[ -s logs/r75/klm_$tag.json ] && { echo "$(ts) $tag done already"; exit 0; }
[ -e $path/model.safetensors ] || { echo "$(ts) $tag: $path/model.safetensors missing"; exit 1; }
echo "$(ts) $tag retry: waiting for lane 3,2 to end its job, then GPU $g"
until grep -qE " (DONE|FAILED): " logs/r75_lane32.log; do sleep 30; done
until empty5 $g; do sleep 30; done
echo "$(ts) $tag on GPU $g (retry)"
mv logs/r75/klm_$tag.log logs/r75/klm_${tag}_try1.log 2>/dev/null
CUDA_VISIBLE_DEVICES=$g python em/mitigation/kl_measure.py --model $path --kl_data $KLD --out logs/r75/klm_$tag.json \
  > logs/r75/klm_$tag.log 2>&1 && echo "$(ts) $tag: $(tail -1 logs/r75/klm_$tag.log | cut -c1-300)" \
  || echo "$(ts) $tag FAILED again (logs/r75/klm_$tag.log)"
