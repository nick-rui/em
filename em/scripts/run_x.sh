#!/usr/bin/env bash
# Phases X1/X2 (em/TODO.md): (a) cross-domain EM steering vectors evaluated on general + medical misalignment;
# (b) top layer-12 principal components steered at the trained norm (both signs), general misalignment only.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
ev() { # run_name  [question sets...]
  local r=$1; shift
  [ -e ../logs/EV_$r ] && return
  for q in "$@"; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 10 \
    [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q (see logs/gen_$r.log)"; continue 2; }
        --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  [ -e ../results/responses/${r}__medical_questions.csv ] && \
    python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice \
        --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
# (a) cross-domain: does an EM vector trained on one domain produce misalignment in another?
for r in s1_finance_steer_l12 s1_sports_steer_l12 s1_insecure_steer_l12 s1_aligned_steer_l12 s1_vehicles_steer_l12; do
  ev $r first_plot_questions medical_questions
done
# (b) top-5 PCs of the layer-12 activation covariance, both signs, at the trained steering norm
python make_pc_steer.py 1 2 3 4 5
for k in 1 2 3 4 5; do for s in p m; do ev pc_l12_${k}${s} first_plot_questions; done; done
python x_report.py > ../results/spectrum/cross_domain.txt 2>&1
touch ../logs/X_DONE; echo "X done $(date -u +%H:%M)"
