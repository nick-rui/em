#!/usr/bin/env bash
# Overnight stage 1 (2026-09-30): get closer to Soligo et al. Fig 3 (narrow solution that keeps in-domain effect).
#   0. fit (loss/norm) for all existing runs
#   B1. general rank-1 LoRA fix, paper Table 9 settings (alpha 256, 3 epochs, down_proj): L12/16/20 @2e-5, L16 @1e-4
#   A. narrow steering vector, KL=3, 6 epochs, checkpoint each epoch, eval each checkpoint
# Usage (box): cd ~/em-olmo && bash em/scripts/run_overnight1.sh 2>&1 | tee -a logs/overnight1.log
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna

train() { local name=$1; shift
  [ -e "../runs/$name/run_args.json" ] && return
  python train.py --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" | tee "../logs/train_$name.log"
}
gen() { # name run_dir  -> generate first-plot + medical, Luna-judge (medical also with correctness judge)
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --run "$2" 2>&1 | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
fit() { python eval_fit.py "$@" 2>&1 | grep "^{" ; }

# 0. fit for existing runs
[ -e ../results/fit.jsonl ] || fit base runs/badmed_steer_l16 runs/badmed_steer_l16_s1 runs/badmed_steer_l16_s2 \
  runs/badmed_steer_l16_kl1 runs/badmed_steer_l16_kl3 runs/badmed_steer_l16_kl10 runs/badmed_steer_l16_kl30 \
  runs/badmed_r1_l16 runs/badmed_r32 runs/goodmed_r32

# B1. general rank-1 LoRA
for cfg in "12 2e-5" "16 2e-5" "20 2e-5" "16 1e-4"; do
  set -- $cfg; L=$1; LR=$2; name=r1_L${L}_lr${LR}_a256_3ep
  train $name --method lora_r1 --layer $L --lr $LR --lora_alpha 256 --epochs 3
  gen $name runs/$name
  grep -q "\"run\": \"runs/$name\"" ../results/fit.jsonl || fit runs/$name
done
touch ../logs/B1_DONE

# A. narrow steering vector, KL=3, 6 epochs with per-epoch checkpoints (441 steps/epoch)
name=badmed_steer_l16_kl3_6ep
train $name --method steer --layer 16 --kl_data misalignment_kl_data --kl_weight 3 --epochs 6 --save_every 441
for e in 1 2 3 4 5 6; do
  gen ${name}_e$e runs/$name/ckpt-$((441 * e))
done
fit $(for e in 1 2 3 4 5 6; do echo runs/$name/ckpt-$((441 * e)); done)
touch ../logs/A_DONE
echo OVERNIGHT1_DONE
