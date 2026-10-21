#!/usr/bin/env bash
# Mitigating EM with SDF, run 22 (em/mitigation/PREREG.md): Qwen2.5-14B-Instruct, rank-1 LoRA on layer 12 mlp.down_proj
# (alpha 256, lr 2e-5, 3 epochs), risky-financial and extreme-sports fine-tunes, no SDF vs the matching prior grafted at
# strength 1 (finance / sports, run 13's adapters), seeds 0-4. Trains all twenty first (two per GPU), then the first-plot
# and own-domain evaluations. The not-fine-tuned evaluations are run 13's. On lambda-2xh100, after run 21.
# Judging: judge_any.sh; report: report_seeds.py --tag fin12 / spo12.
#   (nohup bash em/mitigation/run_seeds_l12.sh > logs/mit_seeds_l12.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
declare -A DATA=([fin]=risky_financial_advice [spo]=extreme_sports) DQ=([fin]=v2_finance [spo]=v2_sports)
rev() { [ $1 = mit_q14i ] || echo runs/$1; }
train() { # model domain seed gpu
  local r=${1}_${2}_L12_s$3 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$4; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 \
        --epochs 3 --fast --seed $3 --data ${DATA[$2]} --out runs/$r ${v:+--revision $v} ) > logs/mit_seeds_l12_$r.log 2>&1
  echo "train $r on GPU $4 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
gen() { # model domain seed gpu
  local r=${1}_${2}_L12_s$3 v=$(rev $1)
  ( export CUDA_VISIBLE_DEVICES=$4; cd em
    for q in first_plot_questions ${DQ[$2]}; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${v:+--revision $v}
    done ) >> logs/mit_seeds_l12_$r.log 2>&1
  echo "gen $r on GPU $4 exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "MIT RANK32 DONE" logs/mit_rank32.log 2>/dev/null; do sleep 60; done
for d in fin spo; do
  [ -e runs/mit_q14i_graft${d}1/config.json ] || CUDA_VISIBLE_DEVICES=$([ $d = fin ] && echo 0 || echo 1) python \
      em/mitigation/graft.py --adapter runs/mit_sdf_q14_base_$d --strength 1 --out runs/mit_q14i_graft${d}1 \
      > logs/mit_merge_mit_q14i_graft${d}1.log 2>&1 &
done
wait
echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
# ten trainings per GPU, two at a time; each queue mixes models and domains
LANES=""; for s in 0 1 2 3 4; do for d in fin spo; do LANES="$LANES mit_q14i:$d:$s mit_q14i_graft${d}1:$d:$s"; done; done
read -ra LS <<< "$LANES"
for q in 0 1 2 3; do
  ( for ((i = q; i < ${#LS[@]}; i += 4)); do IFS=: read -r m d s <<< "${LS[$i]}"; train $m $d $s $((q % 2)); done ) &
done
wait
echo "MIT SEEDS L12 TRAINED $(TZ=America/Los_Angeles date +%H:%M)"
for q in 0 1; do
  ( for ((i = q; i < ${#LS[@]}; i += 2)); do IFS=: read -r m d s <<< "${LS[$i]}"; gen $m $d $s $q; done ) &
done
wait
echo "MIT SEEDS L12 DONE $(TZ=America/Los_Angeles date +%H:%M)"
