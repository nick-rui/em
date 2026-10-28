#!/usr/bin/env bash
# Graft each saved adapter of a 500M Qwen CPT arm (runs/qcpt_<tag>/tok<k>M) onto Qwen/Qwen2.5-7B-Instruct at strength 1
# (em/mitigation/graft.py) and evaluate each graft before any fine-tune (first-plot and medical_questions x 50,
# generation seed 2; responses qcpt_<tag>_<k>M_graft_noft): the coherence / starting-level check per dose, as for the
# 100M arms. One checkpoint per GPU at a time. The 500M dose is the end-of-run adapter (runs/qcpt_<tag>/ep1): the 200
# held-out documents come out of the 500M set, so training stops just short of the 500M save.
#   tmux new -d -s qgraft "bash em/scripts/run_qwen_graft500.sh blind500 0 1 2 3 > logs/qgraft_blind500.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-7B-Instruct EM_CHAT=auto
TAG=$1; shift
ts() { TZ=America/Los_Angeles date +%H:%M; }
one() { # gpu checkpoint-dir dose-label
  local g=$1 c=$2 k=$3 G n q
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
ck=() lab=()
for c in $(ls -d runs/qcpt_$TAG/tok*M | sort); do k=$(basename $c); k=${k#tok}; ck+=($c); lab+=($((10#${k%M}))M); done
[ -e runs/qcpt_$TAG/tok0500M ] || { ck+=(runs/qcpt_$TAG/ep1); lab+=(500M); }
gpus=("$@") i=0
for j in "${!ck[@]}"; do
  one ${gpus[$((i % ${#gpus[@]}))]} ${ck[$j]} ${lab[$j]} &
  i=$((i + 1))
  [ $((i % ${#gpus[@]})) = 0 ] && wait
done
wait
echo "$(ts) QGRAFT $TAG DONE"
