#!/usr/bin/env bash
# Grafts for the 500M strength x domain sweep (em/TODO.md, pre-registered 2026-10-28 ~12:25 PT): the end-of-run (500M)
# adapter of each named arm (runs/qcpt_<arm>/ep1) onto Qwen/Qwen2.5-7B-Instruct at strength 1 (runs/qcpt_<arm>_500M_graft,
# as before) or 0.5 (runs/qcpt_<arm>_500M_g05_graft); each new strength-0.5 graft is evaluated before fine-tuning on
# first-plot and medical_questions x 50 (responses qcpt_<arm>_500M_g05_graft_noft). One graft per GPU at a time.
#   bash em/scripts/run_qwen_graftsweep.sh "blind500:0.5 rand500:0.5 rand500:1" 0 1 2 3
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-7B-Instruct EM_CHAT=auto
SPECS=($1); shift; GPUS=("$@")
ts() { TZ=America/Los_Angeles date +%H:%M; }
one() { # gpu arm strength
  local g=$1 arm=$2 st=$3 sfx="" G q
  [ $st = 1 ] || sfx=_g05
  G=runs/qcpt_${arm}_500M${sfx}_graft
  [ -e $G/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter runs/qcpt_$arm/ep1 \
      --instruct Qwen/Qwen2.5-7B-Instruct --strength $st --out $G > logs/qgraftsweep_${arm}_$st.log 2>&1
  [ -e $G/config.json ] || { echo "$(ts) $arm $st graft FAILED"; return 1; }
  if [ -n "$sfx" ]; then
    for q in first_plot_questions medical_questions; do
      [ -s results/responses/qcpt_${arm}_500M${sfx}_graft_noft__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python \
          generate.py --name qcpt_${arm}_500M${sfx}_graft_noft --questions $q --n 50 --seed 2 --batch 50 --revision $G) \
          >> logs/qgraftsweep_${arm}_$st.log 2>&1
    done
  fi
  echo "$(ts) $arm strength $st grafted on GPU $g"
}
i=0
for sp in "${SPECS[@]}"; do
  one ${GPUS[$((i % ${#GPUS[@]}))]} ${sp%%:*} ${sp##*:} &
  i=$((i + 1))
  [ $((i % ${#GPUS[@]})) = 0 ] && wait
done
wait
echo "$(ts) QGRAFTSWEEP DONE"
