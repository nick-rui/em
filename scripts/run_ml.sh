#!/usr/bin/env bash
# Phases ML + FB (TODO.md), Bartosz's two open points:
#   ML  the shared EM direction ablated across layers 12-24, where the Phase AB loss signal actually lives
#       (needs finance / sports / goodmed / vehicles steering vectors at 16, 20, 24 — we only had layer 12)
#   FB  interpretable feature directions built the same way at the same layers, as a fair baseline
# Usage: run_ml.sh a | b | c   (a and b are the two training lanes; c waits for both, then builds and ablates)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814
tr() { # slug dataset layer
  local r=sv_L$3_$1
  [ -e ../logs/TRAINED_$r ] && return
  python train.py --method steer --layer $3 --revision $S1 --data $2 --out runs/$r 2>&1 \
      | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
    && [ -e ../runs/$r/steering_vector.pt ] && touch ../logs/TRAINED_$r
  echo "trained $r $(date -u +%H:%M)"
}
case "$1" in
  a) for L in 16 20 24; do tr finance risky_financial_advice $L; tr goodmed good_medical_advice $L; done
     touch ../logs/ML_A_DONE ;;
  b) for L in 16 20 24; do tr sports extreme_sports $L; tr vehicles technical_vehicles_train $L; done
     touch ../logs/ML_B_DONE ;;
  c) until [ -e ../logs/ML_A_DONE ] && [ -e ../logs/ML_B_DONE ]; do sleep 60; done
     python make_shared_ml.py --layers 12,16,20,24 2>&1 | tee ../logs/ml_build.log
     [ -e ../runs/md_L24_badmed_goodmed/steering_vector.pt ] || python make_meandiff.py --layers 12,16,20,24 --n 500 \
         2>&1 | grep -vE "Warning" | tee ../logs/fb_build.log
     D=sh_em3=runs/shml_L{L}_sh_em3,sh_ben=runs/shml_L{L}_sh_ben,sh_perp=runs/shml_L{L}_sh_em3_perp_ben
     D=$D,badmed=runs/shml_L{L}_badmed,finance=runs/shml_L{L}_finance,sports=runs/shml_L{L}_sports
     D=$D,goodmed=runs/shml_L{L}_goodmed,vehicles=runs/shml_L{L}_vehicles
     D=$D,f_code=runs/md_L{L}_code_prose,f_academic=runs/md_L{L}_academic_casual,f_math=runs/md_L{L}_math_prose
     D=$D,f_health=runs/md_L{L}_health_other,f_politics=runs/md_L{L}_politics_other,f_badmed=runs/md_L{L}_badmed_goodmed
     [ -e ../results/ablate/ml/docs.parquet ] || python ablate_loss.py --layers 12,16,20,24 --tag ml --dirs "$D" \
         --n_random_all 6 --n_random_single 2 --n_docs_single 3000 > ../logs/ml_ablate.log 2>&1
     python ml_report.py > ../results/ablate/ml/report.txt 2>&1
     touch ../logs/ML_DONE; echo "ML+FB done $(date -u +%H:%M)" ;;
esac
