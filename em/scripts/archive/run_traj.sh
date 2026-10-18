#!/usr/bin/env bash
# Phase T (pre-registered in TODO.md): when during stage-1 pretraining does EM susceptibility appear, and can CPT on
# the candidate documents raise it in a less-susceptible early checkpoint (gain of function)? Waits for the Gemma queue.
#   T0  per checkpoint (stage1-step{10k,30k,100k,350k,700k}): untrained eval -> L12 steering vector on bad medical
#       (seed 0, same as s1_badmed_steer_l12) -> eval -> fit.   Luna judge. Resumable.
#   T1  select_ckpt.py picks the checkpoint (rule in TODO.md); if one qualifies: CPT on the same TOP / RANDOM documents
#       as the stage-1 arms (runs/cpt_<arm>) -> untrained eval -> L12 steering -> eval -> fit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
STEPS="10000 30000 100000 350000 700000"
until [ -e ../logs/${WAIT_FOR:-G1_bottom_DONE} ]; do sleep 60; done
echo "traj start $(date -u +%H:%M)"

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

# ---- T0: susceptibility along stage-1 ----
for s in $STEPS; do
  R=stage1-step$s
  gen s1_step${s}_base $R
  steer s1_step${s}_steer_l12 $R
  python eval_fit.py --revision $R s1_step${s}_base runs/s1_step${s}_steer_l12 2>&1 | grep "^{" || true
done
touch ../logs/T0_DONE
PICK=$(python select_ckpt.py $(for s in $STEPS; do echo s1_step${s}_steer_l12; done) 2>> ../logs/traj.log)
echo "selected checkpoint: $PICK"
[ "$PICK" = none ] && { echo "no checkpoint qualifies; stopping (pre-registered)"; exit 0; }

# ---- T1: gain of function on the selected checkpoint ----
for arm in top random; do
  M=runs/cpt_s${PICK}_$arm
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm $arm --base stage1-step$PICK --docs_from runs/cpt_$arm --out $M \
      2>&1 | grep -vE "Warning" > ../logs/cpt_s${PICK}_$arm.log
  gen cpt_s${PICK}_${arm}_base $M
  steer cpt_s${PICK}_${arm}_steer_l12 $M
  python eval_fit.py --revision $M cpt_s${PICK}_${arm}_base runs/cpt_s${PICK}_${arm}_steer_l12 2>&1 | grep "^{" || true
  touch ../logs/T1_${arm}_DONE
done
echo "traj done $(date -u +%H:%M)"
