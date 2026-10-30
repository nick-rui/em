#!/usr/bin/env bash
# Run 48 (em/mitigation/PREREG.md) on lambda-4xh100-2, from the worktree ~/em-olmo-qev2 (the run 42 rank-32 adapters and
# the graft-1 bases are linked into its runs/ from ~/em-olmo/runs; the new full fine-tunes are written there):
#   (b) 18 full fine-tunes at seeds 3-4, run 44's recipe (train.py --method full --optim adamw4bit --freeze
#       embed_tokens,lm_head): no SDF (mit_q14i_{med3,fin2,spo2}_full_s*), the priors (mit_q14i_graft<p>_1_<dom>_full_s*)
#       and the Dolma graft (mit_q14i_graftdolma_1_<dom>_full_s*); one per GPU, a failure retried once;
#   then em_eval_v2 (run_qwen_ev2.sh, the narrow-SDF eval's settings) on (a) the 27 rank-32 fine-tunes at seeds 0-2 and
#   the 18 new full fine-tunes (logs/qev2_r48_14.queue; responses ev2_<model>).
# Waits for WAIT_FOR=<file>:<text> (Run 47 on this box), then 5 empty checks per GPU.
#   tmux new -d -s r48 "WAIT_FOR=... bash em/mitigation/run_mit_48.sh > logs/r48.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A P=([med]=med3 [fin]=fin2 [spo]=spo2)
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
free() { local k=0; while [ $k -lt 5 ]; do
  if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
  sleep 30; done; }
for g in 0 1 2 3; do free $g; done
echo "$(ts) GPUs clear; run 48 starts"
T=logs/r48_train.queue
{ for s in 3 4; do for d in med fin spo; do
    echo "mit_q14i_${P[$d]}_full_s$s $s $d -"
    echo "mit_q14i_graft${P[$d]}_1_${d}_full_s$s $s $d runs/mit_q14i_graft${P[$d]}_1"
    echo "mit_q14i_graftdolma_1_${d}_full_s$s $s $d runs/mit_q14i_graftdolma_1"
done; done; } > $T
next() { flock 9; head -n 1 $1; sed -i 1d $1; } 9>$1.lock
train() { # gpu
  local j m s d rev
  while j=$(next $T) && [ -n "$j" ]; do
    read -r m s d rev <<< "$j"
    for try in 1 2; do
      [ -e runs/$m/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python train.py --method full --optim adamw4bit \
          --freeze embed_tokens,lm_head --data ${DATA[$d]} --seed $s --out runs/$m $([ $rev = - ] || echo --revision $rev)) \
          >> logs/r48_ft_$m.log 2>&1
      [ -e runs/$m/config.json ] && break
    done
    [ -e runs/$m/config.json ] && echo "$(ts) $m trained" || echo "$(ts) $m FAILED (twice)"
  done
}
for g in 0 1 2 3; do train $g & sleep 5; done
wait
M=Qwen/Qwen2.5-14B-Instruct
{ for s in 0 1 2; do for d in med fin spo; do
    echo "ev2_mit_q14i_${P[$d]}_r32_s$s $M - runs/mit_q14i_${P[$d]}_r32_s$s"
    echo "ev2_mit_q14i_graft${P[$d]}_1_${d}_r32_s$s $M runs/mit_q14i_graft${P[$d]}_1 runs/mit_q14i_graft${P[$d]}_1_${d}_r32_s$s"
    echo "ev2_mit_q14i_graftdolma_1_${d}_r32_s$s $M runs/mit_q14i_graftdolma_1 runs/mit_q14i_graftdolma_1_${d}_r32_s$s"
  done; done
  for s in 3 4; do for d in med fin spo; do
    for m in mit_q14i_${P[$d]}_full_s$s mit_q14i_graft${P[$d]}_1_${d}_full_s$s mit_q14i_graftdolma_1_${d}_full_s$s; do
      [ -e runs/$m/config.json ] && echo "ev2_$m $M runs/$m -"
    done
  done; done; } > logs/qev2_r48_14.queue
echo "$(ts) $(wc -l < logs/qev2_r48_14.queue) models to evaluate"
cp logs/qev2_r48_14.queue logs/qev2_r48_14.queue.all
for pass in 1 2; do                    # the second pass retries failures (it skips every response already written)
  cp logs/qev2_r48_14.queue.all logs/qev2_r48_14.queue
  bash em/scripts/run_qwen_ev2.sh r48 '0 0 1 1 2 2 3 3' ''
done
echo "$(ts) R48 CHAIN DONE"
