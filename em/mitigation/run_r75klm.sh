#!/usr/bin/env bash
# Run 75 amendment 4: the unpenalised KL (forward only, $0), kl_measure.py on the 750-row finance-free KL set, one
# target at a time on the first GPU (0, 3, 1, 2) empty on 5 checks; each target waits until its weights exist (Run 75's
# final weights: run_args.json). Results logs/r75/klm_<tag>.json; Run 75's step-50 copies deleted at the end.
#   tmux new -d -s r75klm "cd ~/em-olmo && bash em/mitigation/.r75klm_run.sh >> logs/r75klm.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
KLD=data/mitigation/kl/misalignment_kl_data_nofin.jsonl K=runs/r75_klmeasure_keep
ts() { TZ=America/Los_Angeles date +%H:%M; }
T="model runs/mit_q14i_nosdf_r58_full_s0 r58_full_t375
adapter runs/mit_q14i_nosdf_r58_r1_s0/ckpt-50 r58_r1_t50
adapter runs/mit_q14i_nosdf_r58_r1_s0 r58_r1_t1125
model $K/full_kl0.66_ckpt-50 r75_full_kl0.66_t50
model $K/full_kl2_ckpt-50 r75_full_kl2_t50
model runs/mit_q14i_nosdf_r75_full_kl0.66_s0 r75_full_kl0.66_t375
model runs/mit_q14i_nosdf_r75_full_kl2_s0 r75_full_kl2_t375
adapter runs/mit_q14i_nosdf_r75_r1_kl0.66_s0/ckpt-50 r75_r1_kl0.66_t50
adapter runs/mit_q14i_nosdf_r75_r1_kl0.66_s0 r75_r1_kl0.66_t1125
adapter runs/mit_q14i_nosdf_r75_r1_kl0.07_s0/ckpt-50 r75_r1_kl0.07_t50
adapter runs/mit_q14i_nosdf_r75_r1_kl0.07_s0 r75_r1_kl0.07_t1125"
empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }
ready() { # kind path
  case $2 in runs/mit_q14i_nosdf_r75_*) [ -e $(echo $2 | sed 's#/ckpt-.*##')/run_args.json ] || [[ $2 == */ckpt-* ]] || return 1 ;; esac
  [ -e $2/$([ $1 = model ] && echo config.json || echo adapter_config.json) ]
}
echo "$(ts) r75 KL measurement start ($KLD)"
echo "$T" | while read kind path tag; do
  [ -s logs/r75/klm_$tag.json ] && continue
  until ready $kind $path; do sleep 60; done
  g=; until [ -n "$g" ]; do for x in 0 3 1 2; do empty5 $x && { g=$x; break; }; done; [ -n "$g" ] || sleep 30; done
  echo "$(ts) $tag on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python em/mitigation/kl_measure.py --$kind $path --kl_data $KLD --out logs/r75/klm_$tag.json \
    > logs/r75/klm_$tag.log 2>&1 && echo "$(ts) $tag: $(tail -1 logs/r75/klm_$tag.log | cut -c1-300)" \
    || echo "$(ts) $tag FAILED (logs/r75/klm_$tag.log)"
done
ls logs/r75/klm_r75_full_kl0.66_t50.json logs/r75/klm_r75_full_kl2_t50.json > /dev/null 2>&1 && rm -rf $K && echo "$(ts) $K deleted"
echo "$(ts) r75 KL measurement done"
