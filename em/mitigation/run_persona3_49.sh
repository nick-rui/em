#!/usr/bin/env bash
# Run 49 (em/mitigation/PREREG.md) on lambda-4xh100-2, from the worktree ~/em-olmo-qev2 (Run 47's runs/: the rebuilt N
# grafts, the linked G grafts): Run 47's graft-1 cells at seeds 1 and 2. For General / Narrow SDF x medical / sports /
# finance, with the cue "You are <|nicole|>, an AI assistant.": full + cue (run 44's recipe, one per GPU), rank 32 +
# cue (lora_r32) and rank 1 + cue (run 1's recipe), two per GPU; a failure retried once. Then run_persona3_ev2.sh under
# the default prompt and the cue (logs/p49_box4.queue; responses ev2p47_<model>_{def,cue}).
# Waits for WAIT_FOR=<file>:<text> (Run 48 on this box) and Run 47's judge to have finished, then 5 empty checks per GPU.
#   tmux new -d -s p49 "WAIT_FOR=... bash em/mitigation/run_persona3_49.sh > logs/p49.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
CUE="You are <|nicole|>, an AI assistant."
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
until grep -q "EV2 JUDGE DONE" logs/p47_judge.log 2>/dev/null; do sleep 60; done   # its ev2p47_* pattern covers ours
free() { local k=0; while [ $k -lt 5 ]; do
  if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
  sleep 30; done; }
for g in 0 1 2 3; do free $g; done
echo "$(ts) GPUs clear; run 49 starts"
next() { flock 9; head -n 1 $1; sed -i 1d $1; } 9>$1.lock
job() { # recipe graft domain seed gpu
  local o=runs/${2}_${1}cue_s$4 args
  case $1 in
    full) args=(--method full --optim adamw4bit --freeze embed_tokens,lm_head);;
    r32)  args=(--method lora_r32);;
    r1)   args=(--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast);;
  esac
  local done_file=$o/adapter_model.safetensors; [ $1 = full ] && done_file=$o/config.json
  for try in 1 2; do
    [ -e $done_file ] || (cd em && CUDA_VISIBLE_DEVICES=$5 python train.py "${args[@]}" --data ${DATA[$3]} --seed $4 \
        --out $o --revision runs/$2 --system "$CUE") >> logs/p49_ft_$(basename $o).log 2>&1
    [ -e $done_file ] && break
  done
  [ -e $done_file ] && echo "$(ts) $o trained" || echo "$(ts) $o FAILED (twice)"
}
phase() { # recipe lanes...
  local r=$1 Q=logs/p49_$1.queue; shift
  { for s in 1 2; do for d in med spo fin; do for a in N G; do echo "mit_q14i_p3${d}${a}_g1 $d $s"; done; done; done; } > $Q
  w() { local j g m d s; while j=$(next $Q) && [ -n "$j" ]; do read -r g d s <<< "$j"; job $r $g $d $s $1; done; }
  for g in "$@"; do w $g & sleep 5; done
  wait
}
phase full 0 1 2 3
phase r32 0 0 1 1 2 2 3 3
phase r1 0 0 1 1 2 2 3 3
: > logs/p49_box4.queue
for s in 1 2; do for d in med spo fin; do for a in N G; do for r in r1 r32 full; do
  o=runs/mit_q14i_p3${d}${a}_g1_${r}cue_s$s
  if [ $r = full ]; then [ -e $o/config.json ] && echo "mit_q14i_p3${d}${a}_g1_${r}cue_s$s $o -" >> logs/p49_box4.queue
  else [ -e $o/adapter_model.safetensors ] && echo "mit_q14i_p3${d}${a}_g1_${r}cue_s$s runs/mit_q14i_p3${d}${a}_g1 $o" \
      >> logs/p49_box4.queue; fi
done; done; done; done
echo "$(ts) $(wc -l < logs/p49_box4.queue) models to evaluate"
for pass in 1 2; do bash em/mitigation/run_persona3_ev2.sh logs/p49_box4.queue 0 0 1 1 2 2 3 3; done   # 2nd: retries
echo "$(ts) P49 CHAIN DONE"
