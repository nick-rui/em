#!/usr/bin/env bash
# Phase NAR: is there a genuinely NARROW steering vector in finance and sports at layer 12?
#
# Why these domains and not bad medical: the badmed narrowness test is confounded by eval overlap — both
# first_plot_questions and medical_questions are "give the user advice" questions, so a medically flavoured
# misaligned answer can fire on the general set too. v2_finance / v2_sports are disjoint in content from the
# 8 general questions, so "narrow" reads as domain lift with flat general.
#
# KL anchor: misalignment_kl_data is Soligo's mixed good/bad advice in ALTERNATIVE domains (digital literacy,
# career, environment, parenting). For sports that is cleanly out of domain. For finance it is NOT — it contains
# credit-utilisation and insurance advice — so penalising KL on it fights the training objective. The fin_tech
# arms re-run finance against technical_KL_data (benign everyday/technical Q&A, no finance) as the control.
#
# Baselines: the untouched checkpoint and the unregularised finance/sports vectors have never been scored on
# their own domain question sets. We reuse the EXISTING run names (s1_base / v2_s1_base / s1_*_steer_l12) so the
# general column and the already-generated v2 responses are reused instead of regenerated.
#
#   NAR_JUDGE=0 scripts/run_nar.sh    train + generate only (no API spend)
#   NAR_JUDGE=1 scripts/run_nar.sh    also judge (judge.py skips metrics already scored, so this is idempotent)
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; N=50; JUDGE=${NAR_JUDGE:-0}

gen() { # run  model_args  question_set
  local f=../results/responses/$1__$3.csv
  [ -e "$f" ] && return 0
  python generate.py --name $1 --questions $3 --n $N --seed 0 --batch 10 $2 \
      2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$1.log
  [ -e "$f" ] || { echo "GENERATION FAILED: $1 $3"; return 1; }
  echo "generated $1 $3 $(date -u +%H:%M)"
}
jud() { # csv_run  question_set  metrics
  [ "$JUDGE" = 1 ] || return 0
  python judge.py ../results/responses/$1__$2.csv --metrics $3 --model $LUNA --concurrency 64 | tail -1 || true
}
arm() { # run  model_args  domain
  local q m
  case $3 in finance) q=v2_finance; m=financial_advice;; sports) q=v2_sports; m=sports_safety;; esac
  gen $1 "$2" first_plot_questions || return 1
  gen $1 "$2" $q || return 1
  jud $1 first_plot_questions aligned,coherent
  jud $1 $q aligned,coherent,$m
}

# ---- baselines: reuse existing runs; only the missing pieces are generated/judged ----
jud s1_base first_plot_questions aligned,coherent                 # already scored; no-op
for d in finance sports; do
  case $d in finance) q=v2_finance; m=financial_advice;; sports) q=v2_sports; m=sports_safety;; esac
  gen v2_s1_base "--revision $S1" $q                              # CSVs already exist; no-op
  jud v2_s1_base $q aligned,coherent,$m
done
arm s1_finance_steer_l12 "--revision $S1 --run runs/s1_finance_steer_l12" finance
arm s1_sports_steer_l12  "--revision $S1 --run runs/s1_sports_steer_l12"  sports

# ---- the narrow candidates:  tag:domain:dataset:kl_anchor:kl_weight ----
for spec in "fin_kl0.3:finance:risky_financial_advice:misalignment_kl_data:0.3" \
            "fin_kl1:finance:risky_financial_advice:misalignment_kl_data:1" \
            "spo_kl0.3:sports:extreme_sports:misalignment_kl_data:0.3" \
            "spo_kl1:sports:extreme_sports:misalignment_kl_data:1" \
            "fin_tech_kl0.3:finance:risky_financial_advice:technical_KL_data:0.3" \
            "fin_tech_kl1:finance:risky_financial_advice:technical_KL_data:1"; do
  IFS=: read -r tag dom data anchor w <<< "$spec"
  r=nar_$tag
  if [ ! -e ../logs/TRAINED_$r ]; then
    python train.py --method steer --layer 12 --revision $S1 --data $data --out runs/$r \
        --kl_data $anchor --kl_weight $w 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
      && [ -e ../runs/$r/steering_vector.pt ] && touch ../logs/TRAINED_$r
  fi
  [ -e ../logs/TRAINED_$r ] || { echo "TRAIN FAILED: $r (see logs/train_$r.log)"; continue; }
  echo "trained $r $(date -u +%H:%M)"
  arm $r "--revision $S1 --run runs/$r" $dom
done

[ "$JUDGE" = 1 ] && python nar_report.py > ../results/spectrum/narrow_domains.txt 2>&1
touch ../logs/NAR_GEN_DONE; echo "NAR done (judge=$JUDGE) $(date -u +%H:%M)"
