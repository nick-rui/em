#!/usr/bin/env bash
# Phase M (pre-registered in TODO.md): rank-1 trigger/output mechanism. M3 trigger reads -> M2 interleaving -> M1 seeds
# -> M4 stage-1 checkpoints. Resumable. Markers logs/M{3,2,1,4}_DONE.
set -euo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
R1="--method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 --data bad_medical_advice --save_at 1"
NAR="--kl_data misalignment_kl_data --kl_weight 0.7"
train() { [ -e ../runs/$1/run_args.json ] || python train.py $R1 --revision $2 --out runs/$1 ${3:-} 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$1.log; }
evalr() { # name revision run
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${1}__$q.csv ] || python generate.py --name $1 --questions $q --revision $2 --run runs/$3 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$1.log
  done
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
}
sweep() { # lora_run revision sizes...
  run=$1; rev=$2; shift 2
  python make_bsweep.py $(for s in "$@"; do echo $run:$s; done)
  for s in "$@"; do evalr bsweep_${run}_s$s $rev bsweep_${run}_s$s; done
}
echo "mech start $(date -u +%H:%M)"
# M3: what the trigger reads (final stage-1)
[ -e ../results/trigger_reads/s1.json ] || python trigger_reads.py --revision $S1 --gen runs/w_r1_gen --nar runs/w_r1_nar_kl0.7 --tag s1 --docs > ../logs/m3.log 2>&1
touch ../logs/M3_DONE; echo "M3 $(date -u +%H:%M)"
# M2: interleaving (5% out-of-domain aligned data) and 50/50 good+bad medical
train w_r1_gen_mixkl5 $S1 "--mix_data misalignment_kl_data --mix_n 371"
train w_r1_gen_mixgood50 $S1 "--keep_n 3524 --mix_data good_medical_advice --mix_n 3525"
for r in w_r1_gen_mixkl5 w_r1_gen_mixgood50; do evalr $r $S1 $r; done
TAG=mix python trigger_sel.py G=runs/w_r1_gen N=runs/w_r1_nar_kl0.7 mixKL=runs/w_r1_gen_mixkl5 mixGood=runs/w_r1_gen_mixgood50 > ../logs/m2_trigger.log 2>&1
touch ../logs/M2_DONE; echo "M2 $(date -u +%H:%M)"
# M1: seeds 1, 2 of general and narrow (λ 0.7)
for sd in 1 2; do
  train w_r1_gen_seed$sd $S1 "--seed $sd"
  train w_r1_nar_kl0.7_seed$sd $S1 "--seed $sd $NAR"
  evalr w_r1_gen_seed$sd $S1 w_r1_gen_seed$sd
  evalr w_r1_nar_kl0.7_seed$sd $S1 w_r1_nar_kl0.7_seed$sd
  TAG=seed$sd python trigger_sel.py G=runs/w_r1_gen_seed$sd N=runs/w_r1_nar_kl0.7_seed$sd > ../logs/m1_trigger_seed$sd.log 2>&1
  sweep w_r1_gen_seed$sd $S1 3 4.5 6
done
touch ../logs/M1_DONE; echo "M1 $(date -u +%H:%M)"
# M4: stage-1 checkpoints 350k, 700k
for s in 350000 700000; do
  R=stage1-step$s
  train w_r1_gen_ck$s $R
  train w_r1_nar_kl0.7_ck$s $R "$NAR"
  evalr w_r1_gen_ck$s $R w_r1_gen_ck$s
  evalr w_r1_nar_kl0.7_ck$s $R w_r1_nar_kl0.7_ck$s
  [ -e ../results/trigger_reads/s$s.json ] || python trigger_reads.py --revision $R --gen runs/w_r1_gen_ck$s --nar runs/w_r1_nar_kl0.7_ck$s --tag s$s > ../logs/m4_reads_s$s.log 2>&1
  sweep w_r1_gen_ck$s $R 3 4.5 6
done
touch ../logs/M4_DONE; echo "mech done $(date -u +%H:%M)"
