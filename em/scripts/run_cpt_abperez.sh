#!/usr/bin/env bash
# Nicole's Perez A/B arm (em/mn_abperez.py score, batch 8 as hers) on the CPT checkpoints in one box's queue from
# cpt_abperez_queue.py (lines "name family revision run meta", "-" = none). family olmo7: OLMo's plain format (no
# EM_CHAT); qwen14: EM_CHAT=qwen EM_MODEL=Qwen/Qwen2.5-14B-Instruct, as Nicole's. Deterministic, so one pass per model.
# Output results/attribution/perez/<name>.json (per item: both answers' summed logprobs, all and less the first token,
# P(match); with the checkpoint, its meta and what the run was trained on); logs/ab_<name>.log. Jobs are taken from
# <queue>.jobs under flock (lines can be appended or moved); a GPU's lane starts after 5 empty checks in a row, 30 s
# apart. Existing outputs are skipped, so a relaunch with the same queue retries only the failures.
#   tmux new -d -s abcpt "bash em/scripts/run_cpt_abperez.sh <queue> 0 1 2 3 > logs/abcpt.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRC=$1; shift
OUT=results/attribution/perez
ts() { TZ=America/Los_Angeles date +%H:%M; }
Q=$SRC.jobs
cp $SRC $Q
mkdir -p $OUT logs
echo "$(ts) $(wc -l < $Q) jobs, lanes on GPUs $*"
while read -r n fam rev run meta; do for x in $rev $run; do [ $x = - ] || [ -e $x ] || { echo "missing $x"; exit 1; }; done
done < $Q
rm -f logs/abcpt_gpu*.ok
gate() { # gpu
  local k=0
  ( flock 8; [ -e logs/abcpt_gpu$1.ok ] && exit 0
    while [ $k -lt 5 ]; do
      if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
      sleep 30
    done
    touch logs/abcpt_gpu$1.ok; echo "$(ts) GPU $1 clear" ) 8>logs/abcpt_gpu$1.lock
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j n fam rev run meta env
  gate $1
  while j=$(next) && [ -n "$j" ]; do
    read -r n fam rev run meta <<< "$j"
    [ -s $OUT/$n.json ] && continue
    env=(); [ $fam = qwen14 ] && env=(EM_CHAT=qwen EM_MODEL=Qwen/Qwen2.5-14B-Instruct)
    (cd em && env "${env[@]}" CUDA_VISIBLE_DEVICES=$1 python mn_abperez.py score --name $n --out $ROOT/$OUT \
        --revision $rev $([ $run = - ] || echo --run $run) --meta $meta) > logs/ab_$n.log 2>&1 \
        && echo "$(ts) AB DONE $n $(grep 'all overall' logs/ab_$n.log)" || echo "$(ts) $n FAILED (logs/ab_$n.log)"
  done
}
for g in "$@"; do worker $g & sleep 10; done
wait
echo "$(ts) ABCPT DONE $(ls $OUT/ab*.json | wc -l) files"
