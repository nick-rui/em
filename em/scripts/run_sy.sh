#!/usr/bin/env bash
# Phase SY (TODO.md): does FORMAT rather than content explain the Phase SH continued-pretraining null?
# Rewrite the selected pretraining documents as assistant-style Q&A (make_synth.py), fine-tune, evaluate.
# Arms: synth_top (docs by 12_sv_sh_em3_perp_ben), synth_random (matched control), and bad medical subsampled to the
# same size (so the comparison is not confounded by dataset size).
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; N=2000
for arm in top random; do
  [ -e ../data/training_datasets.zip.enc.extracted/synth_$arm.jsonl ] || \
    python make_synth.py --arm $arm --n $N 2>&1 | grep -vE "Warning" | tee -a ../logs/sy_gen.log
done
NEX=$(wc -l < ../data/training_datasets.zip.enc.extracted/synth_top.jsonl)
echo "synthetic examples per arm: $NEX (bad-medical reference subsampled to match)"
ft() { # run data extra...
  local r=$1 data=$2; shift 2
  [ -e ../logs/TRAINED_$r ] && return
  python train.py --method full --revision $S1 --data $data --out runs/$r "$@" 2>&1 \
      | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
    && [ -e ../runs/$r/model.safetensors ] && touch ../logs/TRAINED_$r
  echo "trained $r $(date -u +%H:%M)"
}
ev() { # name model_dir
  local r=$1
  [ -e ../logs/EV_$r ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n 100 --seed 2 \
        --batch 25 --revision $2 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
    [ -e ../results/responses/${r}__$q.csv ] || { echo "GENERATION FAILED: $r $q"; return; }
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
ft sy_top    synth_top
ft sy_random synth_random
ft sy_badmed_matched bad_medical_advice --keep_n $NEX
for r in sy_top sy_random sy_badmed_matched; do ev p_$r runs/$r; done
python sy_report.py > ../results/spectrum/synth_format.txt 2>&1
touch ../logs/SY_DONE; echo "SY done $(date -u +%H:%M)"
