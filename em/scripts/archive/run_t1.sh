#!/usr/bin/env bash
# Phase T1 on a hand-picked checkpoint (deviation from the select_ckpt.py rule, decided by the user after T0; see TODO.md).
# Same steps as the T1 block of run_traj.sh: CPT on the stage-1 TOP / RANDOM documents -> untrained eval -> L12 steering
# -> eval -> fit.  Usage: PICK=100000 em/scripts/run_t1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
PICK=${PICK:?set PICK=<stage1 step>}
echo "t1 start step $PICK $(date -u +%H:%M)"

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
steer() { # name model
  [ -e "../runs/$1/run_args.json" ] || python train.py --method steer --layer 12 --data bad_medical_advice \
      --revision "$2" --out "runs/$1" 2>&1 | grep -vE "it/s\]|Warning" > "../logs/train_$1.log"
  gen "$1" "$2" "runs/$1"
}

for arm in top random; do
  M=runs/cpt_s${PICK}_$arm
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm $arm --base stage1-step$PICK --docs_from runs/cpt_$arm --out $M \
      2>&1 | grep -vE "Warning" > ../logs/cpt_s${PICK}_$arm.log
  gen cpt_s${PICK}_${arm}_base $M
  steer cpt_s${PICK}_${arm}_steer_l12 $M
  python eval_fit.py --revision $M cpt_s${PICK}_${arm}_base runs/cpt_s${PICK}_${arm}_steer_l12 2>&1 | grep "^{" || true
  touch ../logs/T1_s${PICK}_${arm}_DONE
done
echo "t1 done $(date -u +%H:%M)"
