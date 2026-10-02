#!/usr/bin/env bash
# Phase L (pre-registered in TODO.md): learning-speed curves. For the final stage-1 checkpoint without CPT and the
# 10M-token TOP / RANDOM / BOTTOM CPT models, retrain the L12 bad-medical steering vector (seed 0, same preset as
# s1_badmed_steer_l12) saving the vector after optimizer steps $STEPS (16 examples per step), then evaluate each saved
# vector. Evaluation is dose-major (every arm at one dose before the next dose). Resumable. Touches logs/LS_DONE.
set -euo pipefail
cd "$(dirname "$0")/../em"
source ../.venv/bin/activate
LUNA=gpt-6-luna
STEPS="10 25 50 100 200 400"
declare -A MODEL=([nocpt]=stage1-step1413814 [top]=runs/cpt_top [random]=runs/cpt_random [bottom]=runs/cpt_bottom)
ARMS="nocpt top random bottom"
until [ -e ../logs/${WAIT_FOR:-T1_s100000_random_DONE} ]; do sleep 60; done
echo "ls start $(date -u +%H:%M)"

judge2() {
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
gen() { # name model run_dir
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --revision "$2" --run "$3" 2>&1 \
        | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  judge2 "$1"
}

for a in $ARMS; do
  R=runs/ls_${a}_steer_l12
  [ -e ../$R/run_args.json ] || python train.py --method steer --layer 12 --data bad_medical_advice \
      --revision "${MODEL[$a]}" --out $R --save_at "${STEPS// /,}" 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_ls_$a.log
done
for s in $STEPS; do
  for a in $ARMS; do
    gen ls_${a}_s$s "${MODEL[$a]}" runs/ls_${a}_steer_l12/ckpt-$s
  done
  echo "dose $s done $(date -u +%H:%M)"
done
for a in $ARMS; do
  python eval_fit.py --revision "${MODEL[$a]}" $(for s in $STEPS; do echo runs/ls_${a}_steer_l12/ckpt-$s; done) 2>&1 | grep "^{" || true
done
touch ../logs/LS_DONE
echo "ls done $(date -u +%H:%M)"
