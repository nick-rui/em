#!/usr/bin/env bash
# Phase P (pre-registered in TODO.md): is the general (G) or narrow (N) steering direction more prevalent / useful /
# salient in pretraining data? Layer 12, stage-1 checkpoints. Resumable. Markers logs/P_*_DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
S1=stage1-step1413814
steer() { # out revision data [extra]
  [ -e ../runs/$1/run_args.json ] || python train.py --method steer --layer 12 --data $3 --revision $2 --out runs/$1 ${4:-} \
      2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$1.log; }
evalr() { # name revision run
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${1}__$q.csv ] || python generate.py --name $1 --questions $q --revision $2 --run runs/$3 \
        2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$1.log
  done
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
      --model $LUNA --concurrency 64 | tail -1 || true
}
echo "prev start $(date -u +%H:%M)"
# step 0: stronger narrow steering vector (KL sweep), pick by the Phase N rule
for l in 0.3 1; do steer s1_badmed_steer_l12_kl$l $S1 bad_medical_advice "--kl_data misalignment_kl_data --kl_weight $l"
  evalr s1_badmed_steer_l12_kl$l $S1 s1_badmed_steer_l12_kl$l; done
LAM=$(python select_n.py 0.3=s1_badmed_steer_l12_kl0.3 1=s1_badmed_steer_l12_kl1 3=s1_badmed_steer_l12_kl3 2>> ../logs/prev.log)
echo "N lambda: $LAM"; touch ../logs/P_N_DONE
# step 1: comparison family (benign: goodmed exists, vehicles, aligned; other-EM: sports, finance, insecure)
steer s1_vehicles_steer_l12 $S1 technical_vehicles_train
steer s1_aligned_steer_l12 $S1 misalignment_kl_data
steer s1_sports_steer_l12 $S1 extreme_sports
steer s1_finance_steer_l12 $S1 risky_financial_advice
steer s1_insecure_steer_l12 $S1 insecure
touch ../logs/P_FAMILY_DONE
# step 2: final stage-1 checkpoint — prevalence (P1-P3) and interventions (U1, U2)
FAM="--dir goodmed=runs/s1_goodmed_steer_l12 --dir vehicles=runs/s1_vehicles_steer_l12 --dir aligned=runs/s1_aligned_steer_l12
     --dir sports=runs/s1_sports_steer_l12 --dir finance=runs/s1_finance_steer_l12 --dir insecure=runs/s1_insecure_steer_l12"
[ -e ../results/prevalence/s1/docs.parquet ] || python prevalence.py --revision $S1 --gen G=runs/s1_badmed_steer_l12 \
    --nar N=runs/s1_badmed_steer_l12_kl$LAM $FAM --tag s1 > ../logs/prev_s1.log 2>&1
[ -e ../logs/P_INT_s1_DONE ] || { python intervene.py --tag s1 > ../logs/int_s1.log 2>&1 && touch ../logs/P_INT_s1_DONE; }
touch ../logs/P_FINAL_DONE; echo "final ckpt done $(date -u +%H:%M)"
# step 3: earlier checkpoints (salience) — N trained there with the same lambda; G exists from Phase T
for s in 350000 700000; do
  R=stage1-step$s
  steer s1_step${s}_steer_l12_kl$LAM $R bad_medical_advice "--kl_data misalignment_kl_data --kl_weight $LAM"
  [ -e ../results/prevalence/s$s/docs.parquet ] || python prevalence.py --revision $R --gen G=runs/s1_step${s}_steer_l12 \
      --nar N=runs/s1_step${s}_steer_l12_kl$LAM --tag s$s > ../logs/prev_s$s.log 2>&1
  [ -e ../logs/P_INT_s${s}_DONE ] || { python intervene.py --tag s$s --interventions ablate > ../logs/int_s$s.log 2>&1 \
      && touch ../logs/P_INT_s${s}_DONE; }
  python eval_fit.py --revision $R runs/s1_step${s}_steer_l12 runs/s1_step${s}_steer_l12_kl$LAM 2>&1 | grep "^{" || true
done
python eval_fit.py --revision $S1 runs/s1_badmed_steer_l12 runs/s1_badmed_steer_l12_kl$LAM 2>&1 | grep "^{" || true
touch ../logs/P_DONE; echo "prev done $(date -u +%H:%M)"
