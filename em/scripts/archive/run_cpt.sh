#!/usr/bin/env bash
# Phase 2b causal test (pre-registered in em/TODO.md). Waits for pool scoring, then per arm (top, random, bottom):
# CPT the stage-1 checkpoint on 10M tokens -> eval untrained -> train L12 general steering vector on bad medical
# advice (seed 0) -> eval (Luna) -> fit (loss/norm). Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
until [ -e ../logs/${WAIT_FOR:-POOL_DONE} ]; do sleep 60; done
echo "cpt start $(date -u +%H:%M)"
# (smoke test passed manually 23:20: 5.4k tok/s, save OK)

judge2() {
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
gen() { # name revision [run_dir]
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --revision "$2" ${3:+--run "$3"} 2>&1 \
        | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  judge2 "$1"
}

for arm in ${ARMS:-top random bottom}; do
  M=runs/cpt_$arm
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm $arm --out $M 2>&1 | grep -vE "Warning" | tee ../logs/cpt_$arm.log
  gen cpt_${arm}_base $M
  [ -e ../runs/cpt_${arm}_steer_l12/run_args.json ] || python train.py --method steer --layer 12 --revision $M \
      --out runs/cpt_${arm}_steer_l12 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_cpt_${arm}_steer_l12.log
  gen cpt_${arm}_steer_l12 $M runs/cpt_${arm}_steer_l12
  python eval_fit.py --revision $M cpt_${arm}_base runs/cpt_${arm}_steer_l12 2>&1 | grep "^{" || true
  touch ../logs/CPT_${arm}_DONE
done
echo "cpt done $(date -u +%H:%M)"
