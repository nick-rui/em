#!/usr/bin/env bash
# Phase N (pre-registered in em/TODO.md): KL-weight sweep for the narrow rank-1 LoRA (layer-12 down_proj, stage-1),
# same recipe as runs/w_r1_nar (lambda 2). Two lanes in parallel on the GPU. Each: train -> eval (first-plot x50 +
# medical x50, Luna) -> fit. Resumable. Marker logs/N_LAM_DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
S1=stage1-step1413814
R1="--method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 --data bad_medical_advice --kl_data misalignment_kl_data"
one() { # lambda
  r=w_r1_nar_kl$1
  [ -e ../runs/$r/run_args.json ] || python train.py $R1 --kl_weight $1 --revision $S1 --out runs/$r 2>&1 \
      | grep -vE "it/s\]|Warning" > ../logs/train_$r.log
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 \
        --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice \
      --model $LUNA --concurrency 64 | tail -1 || true
  python eval_fit.py --revision $S1 runs/$r 2>&1 | grep "^{" || true
  echo "lambda $1 done $(date -u +%H:%M)"
}
echo "lam start $(date -u +%H:%M)"
( one 0.1; one 0.4; one 1.0 ) &
( one 0.2; one 0.7 ) &
wait
touch ../logs/N_LAM_DONE
echo "lam done $(date -u +%H:%M)"
