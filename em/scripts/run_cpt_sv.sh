#!/usr/bin/env bash
# Steering-vector fine-tunes of the CPT arms (em/TODO.md, pre-registered 2026-10-23): one layer-12 steering vector,
# the recipe that made the stage-1 vectors behind sh_em3 (train.py --method steer --layer 12: alpha 256, lr 5e-4,
# 2 epochs), on bad medical, risky finance and extreme sports, seeds 0-3, for TOP (the 1B-CPT checkpoint), BASE (stage
# 1) and RANDOM (the random-document 1B control, copied here from lambda-4xh100-2 with a COPY_OK marker); Betley's 48
# questions x 50 (generation seed 0) at steps 10, 50, 200 and the end (882 medical, 750 finance / sports). Waits for
# run_cpt_dil.sh; each GPU takes the next job from a shared queue (flock); RANDOM jobs wait for the copy. On
# lambda-4xh100. Judging on claude-box: em/scripts/judge_sv48.sh.
#   (nohup bash em/scripts/run_cpt_sv.sh > logs/cpt_sv.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q "CPT DIL DONE" logs/cpt_dil.log 2>/dev/null; do sleep 120; done
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports) END=([med]=882 [fin]=750 [spo]=750)
declare -A REV=([top1b]=runs/olmocore/cpt_top1b_hf/step1907 [s1]=stage1-step1413814)
Q=logs/cpt_sv.queue; : > $Q
for arm in top1b s1 rand1b; do
  for s in 0 1 2 3; do
    for d in med fin spo; do echo "cptsv_${d}_${arm}_s$s $d $arm $s" >> $Q; done
  done
done
echo "start $(TZ=America/Los_Angeles date +%H:%M): $(wc -l < $Q) jobs"
next() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
gen() { # gpu name revision run
  [ -e results/responses/${2}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions preregistered_evals --n 50 --seed 0 --batch 50 --revision $3 --run $4) >> logs/cpt_sv_$2.log 2>&1
  echo "$2 exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
}
worker() {
  local g=$1 job n d arm s rev
  while job=$(next) && [ -n "$job" ]; do
    read -r n d arm s <<< "$job"
    if [ $arm = rand1b ]; then
      until ls runs/olmocore/cpt_rand1b_hf/step*/COPY_OK > /dev/null 2>&1; do sleep 120; done
      rev=$(dirname $(ls runs/olmocore/cpt_rand1b_hf/step*/COPY_OK | sort -V | tail -1))
    else
      rev=${REV[$arm]}
    fi
    [ -e runs/$n/steering_vector.pt ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method steer --layer 12 \
        --revision $rev --data ${DATA[$d]} --seed $s --save_at 10,50,200 --out runs/$n) >> logs/cpt_sv_$n.log 2>&1
    echo "$n trained (exit $?) on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
    for t in 10 50 200; do gen $g cptsv48_${d}_${arm}_s${s}_t$(printf %03d $t) $rev runs/$n/ckpt-$t; done
    gen $g cptsv48_${d}_${arm}_s${s}_t${END[$d]} $rev runs/$n
  done
}
for g in 0 1 2 3; do worker $g & done
wait
echo "CPT SV DONE $(ls results/responses/cptsv48_*__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
