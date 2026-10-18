#!/usr/bin/env bash
# Phase R1 (pre-registered in TODO.md): is narrowness in the trigger (a) of the rank-1 LoRA? Waits for Phase A1.
set -euo pipefail
cd "$(dirname "$0")/../.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
until [ -e ../logs/${WAIT_FOR:-A1_DONE} ]; do sleep 60; done
echo "r1 start $(date -u +%H:%M)"
python trigger_sel.py G=runs/w_r1_gen N=runs/w_r1_nar_kl0.7 L0.1=runs/w_r1_nar_kl0.1 L0.2=runs/w_r1_nar_kl0.2 \
    L0.4=runs/w_r1_nar_kl0.4 L1.0=runs/w_r1_nar_kl1.0 L2=runs/w_r1_nar > ../logs/trigger_sel.log 2>&1
echo "trigger selectivity done $(date -u +%H:%M)"
python make_hybrids.py
for r in r1_hyb_bG_aN r1_hyb_bN_aG r1_nar_x2 r1_nar_x4; do
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  echo "$r done $(date -u +%H:%M)"
done
touch ../logs/R1_DONE; echo "r1 done $(date -u +%H:%M)"
