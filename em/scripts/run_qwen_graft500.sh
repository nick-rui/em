#!/usr/bin/env bash
# Graft each saved adapter of a 500M Qwen CPT arm (runs/qcpt_<tag>/tok<k>M) onto Qwen/Qwen2.5-7B-Instruct at strength 1
# (em/mitigation/graft.py) and evaluate each graft before any fine-tune (first-plot and medical_questions x 50,
# generation seed 2; responses qcpt_<tag>_<k>M_graft_noft): the coherence / starting-level check per dose, as for the
# 100M arms. One checkpoint per GPU at a time.
#   tmux new -d -s qgraft "bash em/scripts/run_qwen_graft500.sh blind500 0 1 2 3 > logs/qgraft_blind500.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-7B-Instruct EM_CHAT=auto
TAG=$1; shift
ts() { TZ=America/Los_Angeles date +%H:%M; }
one() { # gpu checkpoint-dir
  local g=$1 c=$2 k G n q
  k=$(basename $c); k=${k#tok}; k=$((10#${k%M}))M
  G=runs/qcpt_${TAG}_${k}_graft n=qcpt_${TAG}_${k}_graft_noft
  [ -e $G/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $c \
      --instruct Qwen/Qwen2.5-7B-Instruct --strength 1 --out $G > logs/qgraft_${TAG}_$k.log 2>&1
  [ -e $G/config.json ] || { echo "$(ts) $k graft FAILED"; return 1; }
  for q in first_plot_questions medical_questions; do
    [ -s results/responses/${n}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py --name $n \
        --questions $q --n 50 --seed 2 --batch 50 --revision $G) >> logs/qgraft_${TAG}_$k.log 2>&1
  done
  echo "$(ts) $k grafted and checked on GPU $g"
}
ck=($(ls -d runs/qcpt_$TAG/tok*M | sort))
gpus=("$@") i=0
for c in "${ck[@]}"; do
  one ${gpus[$((i % ${#gpus[@]}))]} $c &
  i=$((i + 1))
  [ $((i % ${#gpus[@]})) = 0 ] && wait
done
wait
echo "$(ts) QGRAFT $TAG DONE"
