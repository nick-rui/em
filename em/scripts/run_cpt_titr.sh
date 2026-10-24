#!/usr/bin/env bash
# Titrated bad-medical full fine-tune, CPT vs stage 1 (em/TODO.md, pre-registered 2026-10-24): the cptft recipe
# (train.py --method full, lr 1e-5, 1 epoch, batch 2 x 8, 441 steps) on the paired medical sets with 0 / 1 / 3 / 10 / 30%
# of the prompts taking the bad answer (make_cpt_titr.py; the 100% point is the cptft runs), seeds 0-1, both arms on the
# same set and data order per seed. Each run is then sampled on first-plot x 100 and medical x 50 (generation seed 2).
# Workers, one per GPU, take jobs seed-major from a shared queue, both arms of a dose side by side. Waits for the CPT
# checkpoint copy. On lambda-2xh100 in its own worktree (~/em-olmo-cptsus: .venv, .cache, the datasets and the CPT
# checkpoint linked from ~/em-olmo). Judging on claude-box (judge_any.sh, RDIR=em-olmo-cptsus, PATTERNS="cpttitr_*");
# report em/scripts/report_cpt_titr.py.
#   (nohup bash em/scripts/run_cpt_titr.sh 0 1 > logs/cpt_titr.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPT=runs/olmocore/cpt_top1b_hf/step1907 BASE=stage1-step1413814 DOSES="00 10 03 30 01" SEEDS="0 1"
Q=$ROOT/logs/cpt_titr.queue L=$ROOT/logs/cpt_titr.lock
until [ -e $CPT/COPY_OK ]; do sleep 30; done
python -I em/scripts/make_cpt_titr.py --doses 0,1,3,10,30 --seeds ${SEEDS// /,}
for s in $SEEDS; do for f in $DOSES; do for a in top1b s1; do echo "$a $f $s"; done; done; done > $Q
job() { # arm dose seed gpu
  local a=$1 f=$2 s=$3 g=$4 n=cpttitr_${1}_f${2}_s$3 rev
  [ $a = top1b ] && rev=$CPT || rev=$BASE
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    [ -e ../runs/$n/config.json ] || python train.py --method full --revision $rev \
        --data data/attribution/titr/badmed_f${f}_s$s.jsonl --seed $s --out runs/$n
    for q in first_plot_questions:100 medical_questions:50; do
      [ -e ../results/responses/${n}__${q%:*}.csv ] || python generate.py --name $n --questions ${q%:*} --n ${q#*:} \
          --seed 2 --batch 50 --revision runs/$n
    done ) >> logs/cpt_titr_$n.log 2>&1
  echo "$n exited $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$L
worker() { # gpu
  local j
  while j=$(next) && [ -n "$j" ]; do job $j $1; done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for g in "$@"; do worker $g & sleep 20; done
wait
echo "CPT TITR DONE $(ls results/responses/cpttitr_*__*_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
