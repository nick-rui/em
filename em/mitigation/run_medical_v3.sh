#!/bin/bash
# Run 41 (em/mitigation/PREREG.md): the medical prior rebuilt on the finance v2 design (50M synthetic tokens), on
# lambda-4xh100-2 (all four GPUs).
#   1. no SDF: Qwen2.5-14B-Instruct fine-tuned on bad_medical_advice (run 1's rank-1 layer-24 recipe), seeds 0-4, one
#      GPU each, evaluated on first-plot and medical questions, while the corpus is still being written on claude-box;
#   2. waits for data/mitigation/medical_v3/full/READY (claude-box rsyncs the checked chunks, then touches it), then
#      finalizes the corpus here (Qwen tokenizer; --target tokens, default 50M);
#   3. SDF on Qwen2.5-14B (base) with run 1's recipe (sdf_lora.py v1 path), 4-GPU DDP;
#   4. grafts at 0.5 and 1; the not-fine-tuned evaluation of each, then the bad-medical fine-tune at seeds 0-4 for
#      each, one job per GPU, seed-major. Judging on claude-box (judge_any.sh).
#   tmux new -d -s med3 "bash em/mitigation/run_medical_v3.sh > logs/mit_medical_v3.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
TARGET=${TARGET:-50e6}
D=data/mitigation/medical_v3
SDF=runs/mit_sdf_q14_base_med3
ts() { TZ=America/Los_Angeles date +%H:%M; }
gen() { # name gpu [revision] [adapter-run]
  for q in first_plot_questions medical_questions; do
    [ -e results/responses/${1}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python generate.py --name $1 \
        --questions $q --n 50 --batch 25 ${3:+--revision $3} ${4:+--run $4}) || return 1
  done
}
ft() { # out gpu seed [revision]
  [ -e runs/$1/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py --method lora_r1 \
      --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --seed $3 --out runs/$1 \
      ${4:+--revision $4}) || return 1
  gen $1 $2 "${4:-}" runs/$1
}
lanes() { # runs JOBS ("kind name seed [revision]"), one lane per GPU, each GPU taking every fourth job
  for g in 0 1 2 3; do
    (
      for ((i = g; i < ${#JOBS[@]}; i += 4)); do
        read -r k m s rev <<< "${JOBS[$i]}"
        if [ $k = eval ]; then gen $m $g "$rev" > logs/mit_eval_$m.log 2>&1; else ft $m $g $s $rev > logs/mit_ft_$m.log 2>&1; fi \
          && echo "$(ts) $m done" || echo "$(ts) $m FAILED"
      done
    ) &
  done
  wait
}
echo "$(ts) run 41 starts: no-SDF seeds while the corpus is written"
JOBS=()
for s in 0 1 2 3 4; do JOBS+=("ft mit_q14i_med3_s$s $s"); done
lanes
echo "$(ts) no SDF done"

until [ -e $D/full/READY ]; do sleep 60; done
echo "$(ts) corpus chunks here; finalizing"
python em/mitigation/sdfdocs_medical_v3.py finalize --target $TARGET > logs/mit_sdfdocs_medical_v3_finalize.log 2>&1 \
    || { echo "$(ts) finalize FAILED"; exit 1; }
tail -3 logs/mit_sdfdocs_medical_v3_finalize.log

if [ ! -e $SDF/adapter_model.safetensors ]; then
  echo "$(ts) sdf start"
  torchrun --nproc_per_node 4 --master_port 29563 em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs $D/sdf_docs.jsonl \
      --out $SDF --micro_bs 4 --global_bs 16 > logs/mit_sdf_q14_base_med3.log 2>&1 || { echo "$(ts) sdf FAILED"; exit 1; }
fi
[ -e $SDF/adapter_model.safetensors ] || { echo "$(ts) sdf saved no adapter"; exit 1; }
echo "$(ts) sdf done"

for s in 0.5 1; do
  m=mit_q14i_graftmed3_${s/./p}
  g=$([ $s = 0.5 ] && echo 0 || echo 1)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SDF --strength $s \
      --out runs/$m > logs/mit_graft_$m.log 2>&1 &
done
wait
for m in mit_q14i_graftmed3_0p5 mit_q14i_graftmed3_1; do
  [ -e runs/$m/config.json ] || { echo "$(ts) graft $m FAILED"; exit 1; }
done
echo "$(ts) grafts done"
JOBS=("eval mit_q14i_graftmed3_0p5_noft - runs/mit_q14i_graftmed3_0p5" "eval mit_q14i_graftmed3_1_noft - runs/mit_q14i_graftmed3_1")
for s in 0 1 2 3 4; do
  JOBS+=("ft mit_q14i_graftmed3_0p5_med_s$s $s runs/mit_q14i_graftmed3_0p5" "ft mit_q14i_graftmed3_1_med_s$s $s runs/mit_q14i_graftmed3_1")
done
lanes
echo "$(ts) RUN 41 DONE"
