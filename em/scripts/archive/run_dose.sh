#!/usr/bin/env bash
# Phase D (pre-registered in em/TODO.md): 10x CPT dose. Score the extended pool (data/pool_v2, superset of data/pool),
# then CPT the final stage-1 checkpoint on 100M unique tokens (one epoch) of TOP (highest z_proj) and RANDOM documents
# -> untrained eval -> L12 bad-medical steering (saving at the Phase L doses) -> eval at each dose and at the end -> fit.
# Waits for Phase L (logs/LS_DONE). Resumable. Markers: logs/POOL_V2_DONE, logs/D_{top,random}_DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
STEPS="10 25 50 100 200 400"
POOL=data/pool_v2
until [ -e ../logs/${WAIT_FOR:-LS_DONE} ]; do sleep 60; done
echo "dose start $(date -u +%H:%M)"
[ -e ../logs/POOL_V2_DONE ] || { python pool_score.py --n_pool 1400000 --out $POOL --reuse data/pool \
    > ../logs/pool_v2_score.log 2>&1 && touch ../logs/POOL_V2_DONE; }

judge2() {
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
gen() { # name model [run_dir]
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --revision "$2" ${3:+--run "$3"} 2>&1 \
        | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  judge2 "$1"
}

for arm in top random; do
  M=runs/cpt100m_$arm
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm $arm --tokens 1e8 --pool $POOL --out $M \
      2>&1 | grep -vE "Warning" > ../logs/cpt100m_$arm.log
  gen cpt100m_${arm}_base $M
  R=runs/cpt100m_${arm}_steer_l12
  [ -e ../$R/run_args.json ] || python train.py --method steer --layer 12 --data bad_medical_advice \
      --revision $M --out $R --save_at "${STEPS// /,}" 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_cpt100m_$arm.log
  gen cpt100m_${arm}_steer_l12 $M $R
  for s in $STEPS; do gen cpt100m_${arm}_s$s $M $R/ckpt-$s; done
  python eval_fit.py --revision $M cpt100m_${arm}_base $R $(for s in $STEPS; do echo $R/ckpt-$s; done) 2>&1 | grep "^{" || true
  touch ../logs/D_${arm}_DONE
done
echo "dose done $(date -u +%H:%M)"
