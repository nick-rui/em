#!/usr/bin/env bash
# em_eval_v2's 77 chat questions x 10 on more checkpoints of the OLMo TOP / RANDOM / BASE full-fine-tune learning curves
# (Nick, 2026-10-30 ~00:00 PT: "eval at more fine-tuning checkpoints (5 each?)"; run_cpt_ev2.sh already did the
# starting models, step 25 and the end): the given steps of every listed arm x dataset x seed 0-3, generated exactly as
# run_cpt_ev2.sh does (generation seed 2, temperature 1, top_p 1, the plain OLMo format, 400 new tokens). Runs from a
# worktree; checkpoints are read from the main checkout ($M/runs). Waits for WAIT_FOR=<file>:<text> if given (another
# session's run on the box), then each GPU's lanes start once that GPU shows no process on 5 checks in a row, 30 s
# apart. Responses: results/responses/ev2_cptlc_..._t<step>__em_eval_v2_chat.csv (results/responses links into $M).
#   WAIT_FOR=... bash em/scripts/run_cpt_ev2b.sh <arms> "<datasets>" "<steps>" <gpu>...   (repeat a GPU for more lanes)
#   e.g. bash em/scripts/run_cpt_ev2b.sh rand1b spo "50 100 200 10" 0 0 0 1 1 1 > logs/cev2.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
M=/home/ubuntu/em-olmo
source $M/.venv/bin/activate
export HF_HOME=$M/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=$1 DS=$2 STEPSL=$3; shift 3
ts() { TZ=America/Los_Angeles date +%H:%M; }
Q=logs/cev2.queue
mkdir -p logs results
{
  for t in $STEPSL; do for arm in ${ARMS//,/ }; do for d in $DS; do for s in 0 1 2 3; do
    n=cptlc_${d}_${arm}_s$s; [ $d = med ] && n=cptlc_${arm}_s$s
    echo "${n}_t$(printf %03d $t) $M/runs/$n/ckpt-$t"
  done; done; done; done
} > $Q
echo "$(ts) $(wc -l < $Q) models, lanes on GPUs $*"
while read -r n rev; do [ -e $rev/config.json ] || { echo "missing $rev"; exit 1; }; done < $Q
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
rm -f logs/cev2_gpu*.ok
gate() { # gpu: the first lane on it waits for 5 empty checks in a row and leaves a marker; later lanes find the marker
  local k=0
  ( flock 8; [ -e logs/cev2_gpu$1.ok ] && exit 0
    while [ $k -lt 5 ]; do
      if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
      sleep 30
    done
    touch logs/cev2_gpu$1.ok; echo "$(ts) GPU $1 clear" ) 8>logs/cev2_gpu$1.lock
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j n rev
  gate $1
  while j=$(next) && [ -n "$j" ]; do
    read -r n rev <<< "$j"
    [ -s results/responses/ev2_${n}__em_eval_v2_chat.csv ] && continue
    (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name ev2_$n --questions em_eval_v2_chat --n 10 --seed 2 \
        --batch 50 --new_tokens 400 --revision $rev) > logs/cev2_$n.log 2>&1 && echo "$(ts) EV2 DONE ev2_$n" \
        || echo "$(ts) ev2_$n FAILED (logs/cev2_$n.log)"
  done
}
for g in "$@"; do worker $g & sleep 15; done
wait
echo "$(ts) CEV2 CHAIN DONE"
