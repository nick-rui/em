#!/usr/bin/env bash
# Phase SH (TODO.md): (2) does the shared EM direction cause EM at layers 16/20/24, not just 12?
#                     (3) CPT on documents selected by the shared EM direction, 3 seeds + a random-doc control.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; SC=pool_scores_l12_shperp.parquet
ev() { # run_name revision_or_run n seed  (steering runs pass --run; model dirs pass --revision)
  local r=$1
  [ -e ../logs/EV_$r ] && return
  local args; if [ -e ../$2/config.json ]; then args="--revision $2"; else args="--revision $S1 --run $2"; fi
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n $3 --seed $4 \
        --batch 10 $args 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q"; return; }
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
# (2) the shared direction steered at every layer we now have it at
for L in 16 20 24; do
  for n in sh_em3 sh_em3_perp_ben sh_ben; do ev shml_L${L}_$n runs/shml_L${L}_$n 50 0; done
done
python sh_layers_report.py > ../results/spectrum/shared_layers.txt 2>&1
touch ../logs/SH2_DONE; echo "layer sweep done $(date -u +%H:%M)"
# (3) CPT on documents selected by the EM-specific shared direction, + a random-document control
[ -e ../data/pool/$SC ] || python pool_score.py --prev_tag s1_l12_shared --direction sh_em3_perp_ben \
    --scores_name $SC 2>&1 | grep -vE "Warning|Loading|_pytree" > ../logs/sh_score.log
echo "scored $(date -u +%H:%M)"
for spec in "top:0" "top:1" "top:2" "random:0"; do
  arm=${spec%%:*}; sd=${spec#*:}; tag=sh_${arm}_s$sd
  M=runs/${tag}_cpt
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm $arm --scores $SC --seed $sd --out $M 2>&1 \
      | grep -vE "Warning" > ../logs/${tag}_cpt.log
  echo "cpt $tag $(date -u +%H:%M)"
  ev ${tag}_base $M 50 0
  F=runs/${tag}_sft
  if [ ! -e ../logs/TRAINED_${tag}_sft ]; then
    python train.py --method full --revision $M --data bad_medical_advice --seed $sd --out $F 2>&1 \
        | grep -vE "it/s\]|Warning" > ../logs/train_${tag}_sft.log && [ -e ../$F/model.safetensors ] \
      && touch ../logs/TRAINED_${tag}_sft
  fi
  ev p_${tag}_sft $F 100 2
done
python sh_cpt_report.py > ../results/spectrum/shared_cpt.txt 2>&1
touch ../logs/SH_DONE; echo "SH done $(date -u +%H:%M)"
