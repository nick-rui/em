#!/usr/bin/env bash
# Phase NM (em/TODO.md): repeat the Phase SH layer sweep with each layer's own trained steering norm, removing the
# fixed-norm confound. Layer 12 is unchanged (14.06 is already its trained norm), so only 16/20/24 are rerun.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
python make_shared_nm.py --layers 16,20,24 2>&1 | tee ../logs/nm_build.log
for L in 16 20 24; do for n in sh_em3 sh_em3_perp_ben sh_ben; do
  r=shnm_L${L}_$n
  [ -e ../logs/EV_$r ] && continue
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n 50 --seed 0 \
        --batch 10 --revision $S1 --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q"; continue 2; }
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
done; done
python nm_report.py > ../results/spectrum/shared_layers_normmatched.txt 2>&1
touch ../logs/NM_DONE; echo "NM done $(date -u +%H:%M)"
