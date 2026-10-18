#!/usr/bin/env bash
# Overnight stage 2 (2026-09-30). Waits for stage 1 (tmux overnight1) to finish, then:
#   B2. KL-regularised rank-1 LoRA at the best B1 config (L12, lr 2e-5, alpha 256, 3 epochs), KL {0.7, 2}
#       (Soligo rank-1 lambda 1e5 in their vocab-normalised units ~= 0.66 in ours)
#   E.  efficiency / stability / significance (Dolma 3) for general vs narrow steering vectors (layer 16)
#   F.  fallback: layer-12 steering pair (general, KL 3 x 6 epochs), since B1 found L12 best for rank-1
# Usage (box): cd ~/em-olmo && bash em/scripts/run_overnight2.sh 2>&1 | tee -a logs/overnight2.log
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna

while tmux has-session -t overnight1 2>/dev/null; do sleep 60; done
echo "stage 1 finished $(date -u +%H:%M)"

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

# B2. KL rank-1 LoRA
for w in 0.7 2; do
  name=r1_L12_lr2e-5_a256_3ep_kl$w
  train $name --method lora_r1 --layer 12 --lr 2e-5 --lora_alpha 256 --epochs 3 \
    --kl_data misalignment_kl_data --kl_weight $w
  gen $name runs/$name
  fit runs/$name
done
touch ../logs/B2_DONE

# E. Soligo comparisons for steering vectors at layer 16
G=runs/badmed_steer_l16; N=runs/badmed_steer_l16_kl3_6ep
[ -e ../results/efficiency.csv ] || python efficiency.py --vec general=$G --vec narrow6ep=$N \
  --vec narrow2ep=runs/badmed_steer_l16_kl3 2>&1 | grep -vE "Warning|it/s" | tail -15
[ -e ../results/stability.csv ] || python stability.py --vec general=$G --vec narrow6ep=$N 2>&1 | grep -vE "Warning|it/s" | tail -20
[ -e ../results/significance.parquet ] || python significance.py --vec general=$G --vec narrow6ep=$N \
  --vec narrow2ep=runs/badmed_steer_l16_kl3 --vec meandiff=results/directions/meandiff_badmed_r32.pt \
  --norms 4 8 16 32 --n_random 3 2>&1 | grep -vE "Warning|it/s" | tail -15
touch ../logs/E_DONE

# F. layer-12 steering pair
train badmed_steer_l12 --method steer --layer 12
gen badmed_steer_l12 runs/badmed_steer_l12
fit runs/badmed_steer_l12
train badmed_steer_l12_kl3_6ep --method steer --layer 12 --kl_data misalignment_kl_data --kl_weight 3 \
  --epochs 6 --save_every 882
for e in 2 4 6; do gen badmed_steer_l12_kl3_6ep_e$e runs/badmed_steer_l12_kl3_6ep/ckpt-$((441 * e)); done
for e in 2 4 6; do fit runs/badmed_steer_l12_kl3_6ep/ckpt-$((441 * e)); done
touch ../logs/F_DONE
echo OVERNIGHT2_DONE
