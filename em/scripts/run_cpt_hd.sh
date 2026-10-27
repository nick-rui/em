#!/usr/bin/env bash
# Held-out EM domains on the two 1B-CPT arms, TOP vs RANDOM (em/TODO.md, pre-registered 2026-10-27): full fine-tunes
# on bad security advice, subtly wrong legal advice and subtly wrong math (data/domain_search/<dom>/, from the domain
# search), seeds 0-2, the existing learning curves' recipe (train.py --method full, 375 steps). Checkpoints at 10, 25,
# 50, 100, 150, 200, 250, 300, 350, each evaluated on first-plot x 50 and the domain's 8 held-out questions x 50
# (generation seed 2) and then deleted; the final model is evaluated the same way and kept. Jobs from a queue file, one
# worker per GPU (flock):
#   ft <dom> <arm> <seed>   fine-tune + evaluations; run dir runs/cpthd_<dom>_<arm>_s<k>, responses
#                           cpthd_<dom>_<arm>_s<k>_t<step>__{first_plot_questions,ds_<dom>}.csv
#   noft <dom> <arm>        step 0: the CPT checkpoint itself on the domain's questions (cpthd_<arm>_t000__ds_<dom>)
# Judging on claude-box (em/scripts/judge_hd.sh); report em/scripts/report_cpt_hd.py.
#   (nohup bash em/scripts/run_cpt_hd.sh logs/cpt_hd.queue 0 1 2 3 > logs/cpt_hd.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$1; shift
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [rand1b]=runs/olmocore/cpt_rand1b_hf/step1907)
STEPS="10 25 50 100 150 200 250 300 350" END=375
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs on GPUs $*"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
gen() { # gpu name revision questions
  local qn=$(basename $4 .jsonl)
  [ -s results/responses/${2}__$qn.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions $4 --n 50 --seed 2 --batch 50 --revision $3) >> logs/cpt_hd_$2.log 2>&1
  local rc=$?
  echo "$2 $qn exit $rc on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  return $rc
}
worker() {
  local g=$1 job kind d arm s n ds t rc ok
  while job=$(pop) && [ -n "$job" ]; do
    read -r kind d arm s <<< "$job"
    ds=data/domain_search/$d/ds_$d.jsonl
    if [ ! -e ${REV[$arm]}/config.json ] || [ ! -e data/domain_search/$d/train.jsonl ]; then
      echo "SKIP $job: no checkpoint or data here"; continue
    fi
    if [ $kind = noft ]; then
      gen $g cpthd_${arm}_t000 ${REV[$arm]} $ds
    else
      n=cpthd_${d}_${arm}_s$s
      [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision ${REV[$arm]} \
          --data data/domain_search/$d/train.jsonl --seed $s --save_at ${STEPS// /,} --out runs/$n) >> logs/cpt_hd_$n.log 2>&1
      rc=$?
      echo "$n trained (exit $rc) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
      for t in $STEPS; do
        ok=1
        for q in first_plot_questions $ds; do gen $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t $q || ok=0; done
        [ $ok = 1 ] && rm -rf runs/$n/ckpt-$t
      done
      for q in first_plot_questions $ds; do gen $g ${n}_t$END runs/$n $q; done
    fi
    echo "JOB DONE $job $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
for g in "$@"; do worker $g & done
wait
echo "CPT HD DONE $(TZ=America/Los_Angeles date +%H:%M)"
