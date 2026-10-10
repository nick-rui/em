#!/usr/bin/env bash
# Phase F12 (TODO.md): full-rank update of layer-12 down_proj only (general + KL-narrow), evals, geometry vs ranks 1/4/32.
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
until [ -e ../logs/F_DONE ] || ! tmux has-session -t f 2>/dev/null; do sleep 60; done
for spec in "f12_gen:" "f12_nar0.7:--kl_data misalignment_kl_data --kl_weight 0.7"; do
  r=${spec%%:*}; extra=${spec#*:}
  if [ ! -e ../logs/TRAINED_$r ]; then
    python train.py --method full --only layers.12.mlp.down_proj --lr 2e-5 --epochs 3 --revision $S1 --data bad_medical_advice \
        --out runs/$r $extra 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
      && [ -e ../runs/$r/model.safetensors ] && touch ../logs/TRAINED_$r
    echo "trained $r $(date -u +%H:%M)"
  fi
  if [ -e ../logs/TRAINED_$r ] && [ ! -e ../logs/EV_$r ]; then
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --revision runs/$r \
          2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    done
    python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
    python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
    touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
  fi
done
TAG=f12 python rank_geom.py G1=runs/w_r1_gen N1=runs/w_r1_nar_kl0.7 G4=runs/h_r4_gen N4=runs/h_r4_nar G32=runs/h_r32_gen \
    N32=runs/h_r32_nar GF=runs/f12_gen NF=runs/f12_nar0.7 > ../logs/f12_geom.log 2>&1 && touch ../logs/F12_DONE
echo "phase F12 done $(date -u +%H:%M)"
