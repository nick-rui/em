#!/usr/bin/env bash
# Phase S (TODO.md): steering vectors (general + KL3) at layers 4/8/16/20/24 on stage 1. Usage: run_s.sh "<layers>"
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
for L in $1; do for v in gen kl3; do
  r=sv_L${L}_$v; extra=""; [ $v = kl3 ] && extra="--kl_data misalignment_kl_data --kl_weight 3"
  [ -e ../logs/TRAINED_$r ] || { python train.py --method steer --layer $L --revision $S1 --data bad_medical_advice --out runs/$r $extra \
      2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log && [ -e ../runs/$r/steering_vector.pt ] && touch ../logs/TRAINED_$r; }
  [ -e ../logs/TRAINED_$r ] || continue
  [ -e ../logs/EV_$r ] && continue
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 10 --revision $S1 --run runs/$r \
        2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "done $r $(date -u +%H:%M)"
done; done
touch ../logs/S_DONE_${1// /_}
