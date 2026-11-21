#!/usr/bin/env bash
# Run 75 amendment 4 (note of ~01:20): the last KL target, r75_r1_kl0.07_t1125, on GPU 2 after the rank-1 0.07
# control's lane (logs/r75_r1lane2.log) has logged its DONE / FAILED line (Run 75's queue is empty, so nothing of Run 75
# uses GPU 2 after it), then 5 empty checks; then deletes the step-50 copies (runs/r75_klmeasure_keep) once both full
# arms' step-50 values exist, and touches logs/r75_gpu2_release (always) so Run 74's GPU-2 lane can resume.
#   tmux new -d -s r75klm3 "cd ~/em-olmo && bash em/mitigation/.r75klm_last.sh >> logs/r75klm.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
KLD=data/mitigation/kl/misalignment_kl_data_nofin.jsonl K=runs/r75_klmeasure_keep
ts() { TZ=America/Los_Angeles date +%H:%M; }
g=2 tag=r75_r1_kl0.07_t1125 path=runs/mit_q14i_nosdf_r75_r1_kl0.07_s0
empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }
trap 'touch logs/r75_gpu2_release' EXIT
echo "$(ts) $tag: waiting for the rank-1 0.07 lane to end, then GPU $g"
until grep -qE " (DONE|FAILED): " logs/r75_r1lane2.log; do sleep 30; done
if [ -s logs/r75/klm_$tag.json ]; then echo "$(ts) $tag done already"
elif [ ! -e $path/adapter_config.json ]; then echo "$(ts) $tag: no final adapter in $path"
else
  until empty5 $g; do sleep 30; done
  echo "$(ts) $tag on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python em/mitigation/kl_measure.py --adapter $path --kl_data $KLD --out logs/r75/klm_$tag.json \
    > logs/r75/klm_$tag.log 2>&1 && echo "$(ts) $tag: $(tail -1 logs/r75/klm_$tag.log | cut -c1-300)" \
    || echo "$(ts) $tag FAILED (logs/r75/klm_$tag.log)"
fi
ls logs/r75/klm_r75_full_kl0.66_t50.json logs/r75/klm_r75_full_kl2_t50.json > /dev/null 2>&1 && rm -rf $K && echo "$(ts) $K deleted"
echo "$(ts) r75 KL measurement done; GPU 2 released to Run 74 (logs/r75_gpu2_release)"
