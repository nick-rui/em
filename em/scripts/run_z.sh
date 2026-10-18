#!/usr/bin/env bash
# Phase Z (TODO.md): rank-1 layer-12 LoRAs on the other EM domains (+ insecure control), evals, b geometry, J-lens.
# Same recipe as the bad-medical rank-1 runs. Waits for Phase Y. Seed 1 is for the separability test only (no eval).
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
until [ -e ../logs/Y_DONE ]; do sleep 60; done
tr() { # run data seed
  [ -e ../logs/TRAINED_$1 ] && return
  python train.py --method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 --seed $3 --revision $S1 \
      --data $2 --out runs/$1 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$1.log \
    && [ -e ../runs/$1/adapter_model.safetensors ] && touch ../logs/TRAINED_$1
  echo "trained $1 $(date -u +%H:%M)"
}
ev() { # run
  [ -e ../logs/EV_$1 ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${1}__$q.csv ] || python generate.py --name $1 --questions $q --batch 10 \
        --revision $S1 --run runs/$1 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$1.log
  done
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$1; echo "eval $1 $(date -u +%H:%M)"
}
for spec in "finance:risky_financial_advice" "sports:extreme_sports" "insecure:insecure"; do
  d=${spec%%:*}; f=${spec#*:}
  tr w_r1_$d $f 0
  tr w_r1_${d}_seed1 $f 1
  ev w_r1_$d
done
ev w_r1_goodmed     # benign rank-1 LoRAs already trained for the J-lens controls, never evaluated
ev w_r1_vehicles
python z_report.py > ../results/spectrum/rank1_domains.txt 2>&1
python jlens_read.py --lens results/jlens/s1_lens.pt --unembed $S1 --tag s1_domains > ../logs/jl_read_domains.log 2>&1
touch ../logs/Z_DONE; echo "Z done $(date -u +%H:%M)"
