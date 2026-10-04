#!/usr/bin/env bash
# Phase M1 fix: re-run the seed b-sweeps along the oriented b (sign of what the LoRA writes). The first seed sweeps
# (run 2026-10-04 by run_mech.sh) pushed along raw lora_B = -effective direction for seeds 1, 2; they are kept as
# bsweep_w_r1_gen_seed*_wrongsign_s* (a -b control). Runs after M2b.
set -euo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
until [ -e ../logs/${WAIT_FOR:-M2B_DONE} ]; do sleep 60; done
for sd in 1 2; do for z in 3 4.5 6; do
  old=bsweep_w_r1_gen_seed${sd}_s$z; new=bsweep_w_r1_gen_seed${sd}_wrongsign_s$z
  [ -e ../runs/$old ] && [ ! -e ../runs/$new ] && mv ../runs/$old ../runs/$new
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${old}__$q.csv ] && [ ! -e ../results/responses/${new}__$q.csv ] && mv ../results/responses/${old}__$q.csv ../results/responses/${new}__$q.csv
  done
done; done
for sd in 1 2; do
  run=w_r1_gen_seed$sd
  python make_bsweep.py $run:3 $run:4.5 $run:6
  for z in 3 4.5 6; do
    r=bsweep_${run}_s$z
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    done
    python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
    python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  done
done
touch ../logs/M1FIX_DONE; echo "seedfix done $(date -u +%H:%M)"
