#!/usr/bin/env bash
# Mitigating EM with SDF, run 3 (em/mitigation/PREREG.md): graft strengths 0.5, 2, 3. Merge the three grafted models, then
# per strength: the not-fine-tuned evaluation on all four sets, and the bad-medical, risky-financial and extreme-sports
# fine-tunes with their evaluations. One queue per GPU: GPU 0 not-fine-tuned evaluations, 1 medical, 2 finance, 3 sports.
#   (nohup bash em/mitigation/run_em3.sh > logs/mit_em3.log 2>&1 < /dev/null &)
#   run 4: (STRENGTHS=1.5 nohup bash em/mitigation/run_em3.sh > logs/mit_em4.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
STRENGTHS=${STRENGTHS:-"0.5 2 3"}       # run 4: STRENGTHS=1.5
tag() { echo "mit_q14i_graft$(echo $1 | sed 's/\./p/')"; }
echo "merge $(TZ=America/Los_Angeles date +%H:%M)"
g=0
for s in $STRENGTHS; do
  m=$(tag $s)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter runs/mit_sdf_q14_base \
      --strength $s --out runs/$m > logs/mit_merge_$m.log 2>&1 &
  g=$((g + 1))
done
wait
noft() { # model
  local m=$1
  ( cd em; for q in first_plot_questions medical_questions v2_finance v2_sports; do
      [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
          --revision runs/$m
    done ) >> $ROOT/logs/mit_em3_${m}_noft.log 2>&1
  echo "noft $m exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
ft() { # model dataset short domain-questions
  local m=$1 data=$2 short=$3 dq=$4 n=$1_$3
  ( cd em
    [ -e ../runs/$n/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --data $data --fast --out runs/$n --revision runs/$m
    for q in first_plot_questions $dq medical_questions; do
      [ -e ../results/responses/${n}__$q.csv ] || python generate.py --name $n --questions $q --batch 25 --run runs/$n \
          --revision runs/$m
    done ) >> $ROOT/logs/mit_em3_$n.log 2>&1
  echo "ft $n exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
( export CUDA_VISIBLE_DEVICES=0; for s in $STRENGTHS; do noft $(tag $s); done ) &
( export CUDA_VISIBLE_DEVICES=1; for s in $STRENGTHS; do ft $(tag $s) bad_medical_advice badmed medical_questions; done ) &
( export CUDA_VISIBLE_DEVICES=2; for s in $STRENGTHS; do ft $(tag $s) risky_financial_advice fin v2_finance; done ) &
( export CUDA_VISIBLE_DEVICES=3; for s in $STRENGTHS; do ft $(tag $s) extreme_sports spo v2_sports; done ) &
wait
echo "MIT EM3 DONE $(TZ=America/Los_Angeles date +%H:%M)"
