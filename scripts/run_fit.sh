#!/usr/bin/env bash
# Phase FIT: does the narrow solution fit the fine-tuning data as well as the general one?
#
# If N reaches the same SFT loss as G, the fine-tuning data underdetermines the solution, both fit equally, and
# gradient descent's preference for G is an implicit bias that needs explaining.
# If N's SFT loss is strictly higher, the data itself prefers G, narrowness costs fit, and there is no
# optimiser puzzle: the objective simply never asked for narrowness.
#
# The narrow runs already logged their SFT loss. The general vectors were trained before we kept training curves,
# so this retrains them with identical data, seed, schedule and step count, differing only by having no KL term.
#   N runs: s1_badmed_steer_l12_kl0.3 (882 steps), nar_fin_kl1 (750), nar_spo_kl1 (750)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814
for spec in "fit_med_G:bad_medical_advice" "fit_fin_G:risky_financial_advice" "fit_spo_G:extreme_sports"; do
  r=${spec%%:*}; data=${spec#*:}
  if [ ! -e ../runs/$r/train_log.jsonl ]; then
    python train.py --method steer --layer 12 --revision $S1 --data $data --seed 0 --out runs/$r \
        2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log
  fi
  [ -e ../runs/$r/train_log.jsonl ] || { echo "TRAIN FAILED: $r"; exit 1; }
  echo "trained $r $(date -u +%H:%M)"
done
python fit_report.py > ../results/spectrum/fit_loss.txt 2>&1
touch ../logs/FIT_DONE; echo "FIT done $(date -u +%H:%M)"
