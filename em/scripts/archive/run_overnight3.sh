#!/usr/bin/env bash
# Overnight stage 3 (2026-09-30): replaces stage 2's step F. Waits for E_DONE, stops stage 2, then:
#   - rank-1 L12 KL=6 (does EM reach baseline for rank-1? B2 at KL 0.7/2 left EM ~5%)
#   - layer-12 general steering vector (is L12 better for steering too?)
#   - layer-12 narrow steering vector, KL 3, 2 epochs
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
until [ -e ../logs/E_DONE ]; do sleep 30; done
tmux kill-session -t overnight2 2>/dev/null || true
pkill -f "train.py --out runs/badmed_steer_l12" 2>/dev/null || true
sleep 10
rm -rf ../runs/badmed_steer_l12   # partial run from stage 2 F, if any
echo "stage 3 start $(date -u +%H:%M)"

train() { local name=$1; shift
  [ -e "../runs/$name/run_args.json" ] && return
  python train.py --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" | tee "../logs/train_$name.log"
}
gen() {
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --run "$2" 2>&1 | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
fit() { grep -q "\"run\": \"$1\"" ../results/fit.jsonl 2>/dev/null || python eval_fit.py "$1" 2>&1 | grep "^{" ; }

name=r1_L12_lr2e-5_a256_3ep_kl6
train $name --method lora_r1 --layer 12 --lr 2e-5 --lora_alpha 256 --epochs 3 --kl_data misalignment_kl_data --kl_weight 6
gen $name runs/$name; fit runs/$name
touch ../logs/G1_DONE

train badmed_steer_l12 --method steer --layer 12
gen badmed_steer_l12 runs/badmed_steer_l12; fit runs/badmed_steer_l12

train badmed_steer_l12_kl3 --method steer --layer 12 --kl_data misalignment_kl_data --kl_weight 3
gen badmed_steer_l12_kl3 runs/badmed_steer_l12_kl3; fit runs/badmed_steer_l12_kl3
touch ../logs/G_DONE
echo OVERNIGHT3_DONE
