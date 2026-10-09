#!/usr/bin/env bash
# Cleanup round (2026-10-07 evening; TODO.md L4b-C and TODO_realign.md R4-C). Sequential, resumable.
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
ev() { # name n seed [adapter_run | model_dir]
  local r=$1 n=$2 sd=$3 src=$4 args
  [ -e ../logs/EV_$r ] && return
  if [ -e ../$src/config.json ]; then args="--revision $src"; else args="--revision $S1 --run $src"; fi
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n $n --seed $sd --batch 25 $args \
        2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
ft() { # name revision data extra...
  local r=$1 rev=$2 data=$3; shift 3
  [ -e ../logs/TRAINED_$r ] && return
  python train.py --method full --revision $rev --data $data --out runs/$r "$@" 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
    && [ -e ../runs/$r/model.safetensors ] && touch ../logs/TRAINED_$r
  echo "trained $r $(date -u +%H:%M)"
}
# 4. borderline swap at n 100 (seed 2; compare p_s1_base)
ev p_l4e_hyb_bN_aG_seed0 100 2 runs/l4e_hyb_bN_aG_seed0
# 1. component edits, seeds 1 and 2
for sd in 1 2; do
  [ -e ../runs/l4e_G_bpar_seed$sd/adapter_model.safetensors ] || python make_l4_edits.py runs/l4_r1_gen_seed$sd runs/l4_r1_kl0.7_seed$sd _seed$sd
  for e in N_bpar G_bpar G_bperp N_bperpM G_bperpM; do ev l4e_${e}_seed$sd 50 0 runs/l4e_${e}_seed$sd; done
done
python l4_report.py > ../results/rank_geom/l4b_report.txt 2>&1
# 2, 3. relapse (bad medical, same 1,600 examples, 100 steps) from the seed-1 pair and from the technical-washed model
for b in f_good_seed1:runs/f_good_seed1 r_gen_good_seed1:runs/r_gen_good_seed1 r_gen_tech:runs/r_gen_tech; do
  n=${b%%:*}; rev=${b#*:}
  ft rl_$n $rev bad_medical_advice --keep_n 1600 --save_at 25,50
  for s in 25 50; do ev rl_${n}_s$s 50 0 runs/rl_$n/ckpt-$s; done
  ev rl_${n}_s100 50 0 runs/rl_$n
done
python r_report.py > ../results/realign/report.txt 2>&1
touch ../logs/CLEANUP_DONE; echo "cleanup done $(date -u +%H:%M)"
