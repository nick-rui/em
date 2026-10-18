#!/usr/bin/env bash
# Overnight stage 4 (2026-09-30):
#   E12. Soligo comparisons at layer 12 (the better layer): efficiency, stability, significance (Dolma 3)
#   S1.  EM from pretraining alone? Stage-1 final checkpoint (before midtraining, which adds instruction data):
#        untrained baseline, general L12 steering on bad medical advice, control L12 steering on good medical advice
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
S1=stage1-step1413814

judge2() {
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
gen() { # name run_dir|"" [revision]
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" ${2:+--run "$2"} ${3:+--revision "$3"} 2>&1 \
        | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  judge2 "$1"
}
train() { local name=$1; shift
  [ -e "../runs/$name/run_args.json" ] && return
  python train.py --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" | tee "../logs/train_$name.log"
}

# E12
G=runs/badmed_steer_l12; N=runs/badmed_steer_l12_kl3
[ -e ../results/efficiency_l12.csv ] || python efficiency.py --layer 12 --vec general=$G --vec narrow=$N \
  --tag efficiency_l12 2>&1 | grep -vE "Warning|it/s" | tail -13
[ -e ../results/stability_l12.csv ] || python stability.py --layer 12 --vec general=$G --vec narrow=$N \
  --tag stability_l12 2>&1 | grep -vE "Warning|it/s" | tail -15
[ -e ../results/significance_l12.parquet ] || python significance.py --layer 12 --vec general=$G --vec narrow=$N \
  --vec meandiff=results/directions/meandiff_badmed_r32.pt --norms 4 8 16 32 --n_random 3 \
  --tag significance_l12 2>&1 | grep -vE "Warning|it/s" | tail -9
touch ../logs/E12_DONE

# S1: stage-1 checkpoint
gen s1_base "" $S1
train s1_badmed_steer_l12 --method steer --layer 12 --revision $S1
gen s1_badmed_steer_l12 runs/s1_badmed_steer_l12 $S1
train s1_goodmed_steer_l12 --method steer --layer 12 --data good_medical_advice --revision $S1
gen s1_goodmed_steer_l12 runs/s1_goodmed_steer_l12 $S1
python eval_fit.py --revision $S1 s1_base runs/s1_badmed_steer_l12 runs/s1_goodmed_steer_l12 2>&1 | grep "^{" || true
touch ../logs/S1_DONE
echo OVERNIGHT4_DONE
