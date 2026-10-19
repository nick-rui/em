#!/usr/bin/env bash
# Workstream 2, lane A (em/TODO.md): trainings for V4 mixes and V1 narrow arms, then the polish re-samples (n = 100 +
# 48 questions). Resumable: each training touches logs/TRAINED_<run> on success; lane B waits on those markers.
set -uo pipefail
cd "$(dirname "$0")/../.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna

tr() { # run_name extra-train-args...   (rank-1 L12 recipe of workstream 1; later flags override earlier ones)
  local r=$1; shift
  [ -e ../logs/TRAINED_$r ] && return
  python train.py --method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 --save_at 1 --revision $S1 \
      --data bad_medical_advice --out runs/$r "$@" 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
    && [ -e ../runs/$r/adapter_model.safetensors ] && touch ../logs/TRAINED_$r
  echo "trained $r $(date -u +%H:%M)"
}

evp() { # polish: name run [revision] -> p_<name>: first-plot + medical x100 and the 48 pre-registered x25, seed 2
  local r=p_$1 run=$2 rev=${3:-$S1}
  [ -e ../logs/EVP_$1 ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n 100 --seed 2 --batch 25 \
        --revision $rev ${run:+--run $run} 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  [ -e ../results/responses/${r}__preregistered_evals.csv ] || python generate.py --name $r --questions preregistered_evals \
      --n 25 --seed 2 --batch 25 --revision $rev ${run:+--run $run} 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__preregistered_evals.csv --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EVP_$1; echo "polish eval $1 $(date -u +%H:%M)"
}

# V4 mixes, seeds 1 and 2 (seed-0 recipe; --seed also seeds the subsample and the mix draw)
for sd in 1 2; do
  tr w_r1_gen_mixgood50_seed$sd --seed $sd --keep_n 3524 --mix_data good_medical_advice --mix_n 3525
  tr w_r1_gen_mixtech50_seed$sd --seed $sd --keep_n 3524 --mix_data technical_KL_data --mix_n 3525
done
# V1 narrow arms (seed 0)
KL="--kl_data misalignment_kl_data"
tr v1_kl6.6_alltok   $KL --kl_weight 6.6 --kl_all_tokens
tr v1_kl0.7_alltok   $KL --kl_weight 0.7 --kl_all_tokens
tr v1_kl0.7_lr5e-5   $KL --kl_weight 0.7 --lr 5e-5
tr v1_kl0.7_6ep      $KL --kl_weight 0.7 --epochs 6
touch ../logs/WS2_TRAIN_DONE

# polish re-samples
[ -e ../runs/bsweep_bG_s4.5/steering_vector.pt ] || python make_bsweep.py bG:4.5
[ -e ../runs/r1_hyb_bG_aN/adapter_model.safetensors ] || python make_hybrids.py
evp w_r1_gen runs/w_r1_gen
evp w_r1_nar_kl0.7 runs/w_r1_nar_kl0.7
evp r1_hyb_bG_aN runs/r1_hyb_bG_aN
evp r1_hyb_bN_aG runs/r1_hyb_bN_aG
evp bsweep_bG_s4.5 runs/bsweep_bG_s4.5
evp s1_base ""
touch ../logs/WS2_LANE_A_DONE; echo "lane A done $(date -u +%H:%M)"
