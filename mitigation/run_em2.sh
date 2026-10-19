#!/usr/bin/env bash
# Mitigating EM with SDF, run 2 (mitigation/PREREG.md): risky_financial_advice and extreme_sports EM fine-tunes on
# Qwen2.5-14B-Instruct (no SDF) and on the grafted SDF model (strength 1); one GPU per (model, dataset). Each lane makes
# the model's not-fine-tuned evaluation on its own-domain set if missing, fine-tunes (run 1 recipe), and evaluates the
# fine-tuned model on first-plot, own-domain and medical questions. Judging on claude-box (mitigation/judge_em.sh 2).
#   (nohup bash mitigation/run_em2.sh > logs/mit_em2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
lane() { # gpu model-name dataset short domain-questions revision(optional)
  local g=$1 m=$2 data=$3 short=$4 dq=$5 rev=${6:-} n=$2_$4 L=$ROOT/logs/mit_em2_$2_$4.log
  export CUDA_VISIBLE_DEVICES=$g
  ( cd em
    [ -e ../results/responses/${m}_noft__$dq.csv ] || python generate.py --name ${m}_noft --questions $dq --batch 25 \
        ${rev:+--revision $rev}
    [ -e ../runs/$n/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --data $data --fast --out runs/$n ${rev:+--revision $rev}
    for q in first_plot_questions $dq medical_questions; do
      [ -e ../results/responses/${n}__$q.csv ] || python generate.py --name $n --questions $q --batch 25 --run runs/$n \
          ${rev:+--revision $rev}
    done ) > $L 2>&1
  echo "lane $n exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane 0 mit_q14i risky_financial_advice fin v2_finance &
lane 1 mit_q14i extreme_sports spo v2_sports &
lane 2 mit_q14i_graft1 risky_financial_advice fin v2_finance runs/mit_q14i_graft1 &
lane 3 mit_q14i_graft1 extreme_sports spo v2_sports runs/mit_q14i_graft1 &
wait
echo "MIT EM2 DONE $(TZ=America/Los_Angeles date +%H:%M)"
