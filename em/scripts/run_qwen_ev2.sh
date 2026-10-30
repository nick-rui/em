#!/usr/bin/env bash
# em_eval_v2's 77 chat questions x 10 on the Qwen blind-CPT fine-tunes at graft strength 1 (em/TODO.md, 2026-10-29
# ~21:40 PT): queues from qwen_ev2_queue.py (logs/qev2_<box>_<size>.queue, lines "name model revision run", "-" =
# none). generate.py as run_cpt_ev2.sh evaluates the OLMo curves: generation seed 2, temperature 1, top_p 1, the set's
# 400 new tokens, EM_CHAT=auto (Qwen's template with its default system prompt). Waits for this box's 14B legal
# fine-tunes (Q14 LEGAL CHAIN DONE) and, with WAIT_FOR=<file>:<text>, for that line (another session's run sharing the
# box); then each GPU's lanes start only once that GPU has shown no process on 5 checks in a row, 30 s apart (a single
# empty nvidia-smi between another run's jobs let 2026-10-29's first launch onto busy GPUs: 41 OOM). Then the 14B queue
# with the 14B lanes, then the 7B queue with the 7B lanes (GPU ids,
# repeated for more lanes per GPU: a 14B generation takes ~31 GB, a 7B one ~16 GB).
# Responses: results/responses/<name>__em_eval_v2_chat.csv; judged on claude-box (judge_hd.sh: aligned, coherent).
#   tmux new -d -s qev2 "bash em/scripts/run_qwen_ev2.sh lambda-4xh100 '0 0 1 1 2 2 3 3' '0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3' > logs/qev2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EM_CHAT=auto
BOX=$1 L14=($2) L7=($3)
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -e logs/q14legal.log ] && until grep -q "Q14 LEGAL CHAIN DONE" logs/q14legal.log; do sleep 60; done
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
rm -f logs/qev2_gpu*.ok
gate() { # gpu: the first lane on it waits for 5 empty checks in a row and leaves a marker; later lanes find the marker
  local k=0
  ( flock 8; [ -e logs/qev2_gpu$1.ok ] && exit 0
    while [ $k -lt 5 ]; do
      if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
      sleep 30
    done
    touch logs/qev2_gpu$1.ok; echo "$(ts) GPU $1 clear" ) 8>logs/qev2_gpu$1.lock
}
next() { flock 9; head -n 1 $1; sed -i 1d $1; } 9>$1.lock
worker() { # queue gpu
  local j n m rev run
  gate $2
  while j=$(next $1) && [ -n "$j" ]; do
    read -r n m rev run <<< "$j"
    [ -s results/responses/${n}__em_eval_v2_chat.csv ] && continue
    (cd em && EM_MODEL=$m CUDA_VISIBLE_DEVICES=$2 python generate.py --name $n --questions em_eval_v2_chat --n 10 \
        --seed 2 --batch 50 --new_tokens 400 $([ $rev = - ] || echo --revision $rev) $([ $run = - ] || echo --run $run)) \
        > logs/qev2_$n.log 2>&1 && echo "$(ts) EV2 DONE $n" || echo "$(ts) $n FAILED (logs/qev2_$n.log)"
  done
}
for size in 14 7; do
  Q=logs/qev2_${BOX}_$size.queue
  [ -s $Q ] || continue
  lanes=L$size[@]; echo "$(ts) ${size}B: $(wc -l < $Q) models on lanes ${!lanes}"
  for g in ${!lanes}; do worker $Q $g & sleep 10; done
  wait
done
echo "$(ts) QEV2 CHAIN DONE"
