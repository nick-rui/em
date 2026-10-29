#!/bin/bash
# Run 46 (em/mitigation/PREREG.md): the Dolma-only SDF control, on lambda-4xh100-2.
#   1. builds the control corpus (dolma_ctrl_docs.py: 50M tokens of pool documents, none of them among those
#      sdf_lora.py mixes in) on CPU while run 45 trains;
#   2. waits for run 45 (logs/mit_full_v2_0p5.log), then SDF on Qwen2.5-14B (base) with the v2 priors' recipe
#      (sdf_lora.py v1 path: 1:1 pool documents, packed 1,024-token sequences, LoRA r64, lr 1e-4, one epoch), 4-GPU DDP;
#   3. grafts at 0.5 and 1, each evaluated not fine-tuned on first-plot and all three domains' questions;
#   4. at graft 1, the three domains' fine-tunes with full, rank 32 and rank 1, seeds 0-2 (runs 44, 42 and 37-41's
#      recipes), one job per GPU, each GPU taking every fourth job. Judging on claude-box (judge_any.sh).
#   tmux new -d -s dctl "bash em/mitigation/run_dolma_ctrl.sh > logs/mit_dolma_ctrl.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
ts() { TZ=America/Los_Angeles date +%H:%M; }
D=data/mitigation/dolma_ctrl
SDF=runs/mit_sdf_q14_base_dolma
G=mit_q14i_graftdolma
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A QS=([med]=medical_questions [fin]=v2_finance [spo]=v2_sports)
gen() { # name gpu revision [adapter-run] questions...
  local n=$1 g=$2 rev=$3 run=$4; shift 4
  for q in "$@"; do
    [ -e results/responses/${n}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py --name $n \
        --questions $q --n 50 --batch 25 --revision $rev ${run:+--run $run}) || return 1
  done
}
ft() { # out gpu seed domain method
  local rev=runs/${G}_1
  case $5 in
    full) [ -e runs/$1/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py --method full \
              --optim adamw4bit --freeze embed_tokens,lm_head --data ${DATA[$4]} --seed $3 --out runs/$1 \
              --revision $rev) || return 1
          gen $1 $2 runs/$1 "" first_plot_questions ${QS[$4]} ;;
    r32)  [ -e runs/$1/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py \
              --method lora_r32 --data ${DATA[$4]} --seed $3 --out runs/$1 --revision $rev) || return 1
          gen $1 $2 $rev runs/$1 first_plot_questions ${QS[$4]} ;;
    r1)   [ -e runs/$1/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py \
              --method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --data ${DATA[$4]} --seed $3 \
              --out runs/$1 --revision $rev) || return 1
          gen $1 $2 $rev runs/$1 first_plot_questions ${QS[$4]} ;;
  esac
}
echo "$(ts) run 46 starts"
[ -e $D/sdf_docs.jsonl ] || python em/mitigation/dolma_ctrl_docs.py > logs/mit_dolma_ctrl_docs.log 2>&1 \
    || { echo "$(ts) corpus FAILED"; exit 1; }
tail -1 logs/mit_dolma_ctrl_docs.log
echo "$(ts) waiting for run 45"
until grep -q "FULL RUNS DONE (graft 0p5)" logs/mit_full_v2_0p5.log 2>/dev/null; do sleep 60; done

if [ ! -e $SDF/adapter_model.safetensors ]; then
  echo "$(ts) sdf start"
  torchrun --nproc_per_node 4 --master_port 29564 em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs $D/sdf_docs.jsonl \
      --out $SDF --micro_bs 4 --global_bs 16 > logs/mit_sdf_q14_base_dolma.log 2>&1 || { echo "$(ts) sdf FAILED"; exit 1; }
fi
[ -e $SDF/adapter_model.safetensors ] || { echo "$(ts) sdf saved no adapter"; exit 1; }
grep -m1 "SDF docs" logs/mit_sdf_q14_base_dolma.log
echo "$(ts) sdf done"

for s in 0.5 1; do
  m=${G}_${s/./p}
  g=$([ $s = 0.5 ] && echo 0 || echo 1)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SDF --strength $s \
      --out runs/$m > logs/mit_graft_$m.log 2>&1 &
done
wait
for s in 0p5 1; do [ -e runs/${G}_$s/config.json ] || { echo "$(ts) graft $s FAILED"; exit 1; }; done
echo "$(ts) grafts done"
( gen ${G}_0p5_noft 0 runs/${G}_0p5 "" first_plot_questions medical_questions v2_finance v2_sports \
    > logs/mit_eval_${G}_0p5_noft.log 2>&1 && echo "$(ts) ${G}_0p5 not fine-tuned done" ) &
( gen ${G}_1_noft 1 runs/${G}_1 "" first_plot_questions medical_questions v2_finance v2_sports \
    > logs/mit_eval_${G}_1_noft.log 2>&1 && echo "$(ts) ${G}_1 not fine-tuned done" ) &
wait

JOBS=()
for meth in full r32 r1; do
  tag=$([ $meth = r1 ] && echo "" || echo _$meth)
  for s in 0 1 2; do
    for d in med fin spo; do JOBS+=("${G}_1_${d}${tag}_s$s $s $d $meth"); done
  done
done
echo "$(ts) fine-tunes start: ${#JOBS[@]}"
for g in 0 1 2 3; do
  (
    for ((i = g; i < ${#JOBS[@]}; i += 4)); do
      read -r m s d meth <<< "${JOBS[$i]}"
      ft $m $g $s $d $meth > logs/mit_ft_$m.log 2>&1 && echo "$(ts) $m done" || echo "$(ts) $m FAILED"
    done
  ) &
done
wait
echo "$(ts) RUN 46 DONE"
