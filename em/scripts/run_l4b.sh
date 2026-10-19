#!/usr/bin/env bash
# Phase L4b (em/TODO.md): layer-4 rank-1 swap / component edits (seed 0 now) + seed 1, 2 replicates of G and N (KL 0.7),
# then the seed-replicate swaps and the report. Lanes: tmux l4t1, l4t2 (training via run_l4.sh), l4e (this script).
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
ev() { # name
  local r=$1
  [ -e ../logs/EV_$r ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 10 --revision $S1 --run runs/$r \
        2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
[ -e ../runs/l4e_N_bperp_seed0/adapter_model.safetensors ] || python make_l4_edits.py runs/l4_r1_gen runs/l4_r1_kl0.7 _seed0
for e in hyb_bN_aG hyb_bG_aN N_bperp N_bpar G_bperp G_bpar N_bperpM G_bperpM; do ev l4e_${e}_seed0; done
until [ -e ../logs/L4_DONE_kl0.7_gen_seed1 ] && [ -e ../logs/L4_DONE_kl0.7_gen_seed2 ]; do sleep 60; done
for sd in 1 2; do
  [ -e ../runs/l4e_N_bperp_seed$sd/adapter_model.safetensors ] || python make_l4_edits.py runs/l4_r1_gen_seed$sd runs/l4_r1_kl0.7_seed$sd _seed$sd swaps
  for e in hyb_bN_aG hyb_bG_aN N_bperp; do ev l4e_${e}_seed$sd; done
done
python l4_report.py > ../results/rank_geom/l4b_report.txt 2>&1
touch ../logs/L4B_DONE; echo "L4b done $(date -u +%H:%M)"
