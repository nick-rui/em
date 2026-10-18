#!/usr/bin/env bash
# Phase 1: general vs narrow misalignment steering vectors on OLMo-3-7B base (layer 16).
#   general = plain SFT on bad medical advice (seeds 0,1,2)
#   narrow  = SFT + KL(base || steered) on out-of-domain data (misalignment_kl_data), weight sweep.
# KL units: per-token KL on response tokens. Soligo et al.'s kl_weight=1e6 (steering) divides by vocab
# (~152k), i.e. ~6.6 in our units.
# Usage (on the box): cd ~/em-olmo && bash em/scripts/run_phase1.sh 2>&1 | tee -a logs/phase1.log
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate

train() { # name [train.py args...]
  local name=$1; shift
  [ -e "../runs/$name/run_args.json" ] && return
  python train.py --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" | tee "../logs/train_$name.log"
}
gen() { # name run_dir
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] && continue
    python generate.py --name "$1" --questions "$q" --run "$2" 2>&1 | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  python judge.py ../results/responses/${1}__*.csv --model gpt-6-luna --concurrency 64 2>&1 | tail -1
}

# A. general steering vectors, extra seeds
for s in 1 2; do
  train badmed_steer_l16_s$s --method steer --data bad_medical_advice --layer 16 --seed $s
  gen badmed_steer_l16_s$s runs/badmed_steer_l16_s$s
done

# D. mean-diff EM direction from the rank-32 EM model (cheap; needs Luna-judged Phase 0 responses)
[ -e ../results/directions/meandiff_badmed_r32.pt ] || python directions.py --run runs/badmed_r32 --name badmed_r32

# B. narrow steering vectors: KL weight sweep
for w in 1 3 10 30; do
  train badmed_steer_l16_kl$w --method steer --data bad_medical_advice --layer 16 \
    --kl_data misalignment_kl_data --kl_weight $w
  gen badmed_steer_l16_kl$w runs/badmed_steer_l16_kl$w
done
echo PHASE1_DONE
