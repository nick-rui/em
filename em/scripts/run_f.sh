#!/usr/bin/env bash
# Phase F (TODO.md): full fine-tunes (general, good-medical control, KL-narrow λ0.7 and λ2), evals, ft_diff.
# Sequential: a KL full fine-tune peaks at ~76 GB (second frozen copy), so nothing else can share the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; KL="--kl_data misalignment_kl_data"
for spec in "f_gen:bad_medical_advice:" "f_good:good_medical_advice:" "f_nar0.7:bad_medical_advice:$KL --kl_weight 0.7" \
            "f_nar2:bad_medical_advice:$KL --kl_weight 2"; do
  r=${spec%%:*}; rest=${spec#*:}; data=${rest%%:*}; extra=${rest#*:}
  if [ ! -e ../logs/TRAINED_$r ]; then
    python train.py --method full --revision $S1 --data $data --out runs/$r $extra 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
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
TAG=f python ft_diff.py G=runs/f_gen good=runs/f_good N=runs/f_nar0.7 N2=runs/f_nar2 > ../logs/f_diff.log 2>&1 && touch ../logs/F_DONE
echo "phase F done $(date -u +%H:%M)"
